"""
This is a fusion model that is a combination of two different GNN modes - (HGT & SAGEConvolution)
embedded with anomaly detection that works by detecting false energy and supress it that effectively stabalize the model.
"""


class HeteroProjector(nn.Module):
    """
    Projects heterogeneous node features into a shared embedding space.

    Improvements over previous version:
    - LayerNorm per node type (stabilizes hetero scale mismatch)
    - Non-linearity (GELU)
    - Dropout (regularization at projection boundary)
    """

    def __init__(
        self,
        in_dims: Dict[str, int],
        hidden_dim: int,
        node_types: List[str],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_types = node_types
        self.hidden_dim = hidden_dim

        self.projectors = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(in_dims[ntype], hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for ntype in node_types
        })

    def forward(self, data) -> Dict[str, torch.Tensor]:
        x_dict = {}

        for ntype in self.node_types:
            if ntype not in data.node_types:
                raise RuntimeError(f"Missing node type '{ntype}' in data")

            x = data[ntype].x
            if x is None:
                raise RuntimeError(f"data['{ntype}'].x is None")

            x_dict[ntype] = self.projectors[ntype](x)

        return x_dict

Hetero GCN Encoder

class HeteroSAGEEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        metadata,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.node_types = metadata[0]

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                conv_dict[edge_type] = SAGEConv(
                    in_channels=(hidden_dim, hidden_dim),
                    out_channels=hidden_dim,
                    aggr="mean",
                )

            self.convs.append(HeteroConv(conv_dict, aggr="mean"))

            # One LayerNorm per node type
            self.norms.append(
                nn.ModuleDict({
                    ntype: nn.LayerNorm(hidden_dim)
                    for ntype in self.node_types
                })
            )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Final projection
        self.out_proj = nn.ModuleDict({
            ntype: nn.Linear(hidden_dim, out_dim)
            for ntype in self.node_types
        })

    def forward(self, x_dict, edge_index_dict):
        for i in range(self.num_layers):
            x_new = self.convs[i](x_dict, edge_index_dict)

            out = {}
            for ntype in x_new:
                # residual
                h = x_new[ntype] + x_dict[ntype]

                # norm → activation → dropout
                h = self.norms[i][ntype](h)
                h = self.activation(h)
                h = self.dropout(h)

                out[ntype] = h

            x_dict = out

        # final projection
        z_dict = {
            ntype: self.out_proj[ntype](x)
            for ntype, x in x_dict.items()
        }
        return z_dict

HGT Encoder

class HGTEncoder(nn.Module):
    """Stable HGT encoder aligned with classifier + energy head."""

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.node_types = metadata[0]

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_ch = hidden_dim
            out_ch = hidden_dim if i < num_layers - 1 else out_dim

            self.convs.append(
                HGTConv(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    metadata=metadata,
                    heads=num_heads,
                )
            )

            # LayerNorm per node type
            self.norms.append(
                nn.ModuleDict({
                    ntype: nn.LayerNorm(out_ch) for ntype in self.node_types
                })
            )

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)

            # Normalize per node type
            x_dict = {
                ntype: self.norms[i][ntype](x)
                for ntype, x in x_dict.items()
            }

            # Activation + dropout except last layer
            if i < self.num_layers - 1:
                x_dict = {
                    ntype: self.dropout(self.act(x))
                    for ntype, x in x_dict.items()
                }

        return x_dict

Fusion
- To fuse embeddings from GNN and Transformer

class Fusion(nn.Module):
    """
    Stable fusion of GNN and Transformer embeddings.

    - Normalizes each branch before fusion
    - Uses fixed alpha (safe default)
    """

    def __init__(self, out_dim: int, alpha: float = 0.5):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0,1], got {alpha}")

        self.alpha = alpha
        self.norm_gnn = nn.LayerNorm(out_dim)
        self.norm_tr = nn.LayerNorm(out_dim)

    def forward(self, z_gnn: torch.Tensor, z_transformer: torch.Tensor) -> torch.Tensor:
        if z_gnn.shape != z_transformer.shape:
            raise RuntimeError(
                f"Shape mismatch: GNN {tuple(z_gnn.shape)} vs "
                f"Transformer {tuple(z_transformer.shape)}"
            )

        z_gnn = self.norm_gnn(z_gnn)
        z_tr  = self.norm_tr(z_transformer)

        return self.alpha * z_gnn + (1.0 - self.alpha) * z_tr



class GatedFusion(nn.Module):
    """Gated fusion (optional upgrade from simple linear combination).

    Args:
        out_dim: Dimension of input embeddings
    """

    def __init__(self, out_dim: int):
        super().__init__()
        # Gate learns to weight the two branches
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.Sigmoid()
        )

    def forward(
        self,
        z_gnn: torch.Tensor,
        z_transformer: torch.Tensor,
    ) -> torch.Tensor:
        """Gated fusion of GNN and Transformer embeddings.

        Args:
            z_gnn: GNN embeddings [N, out_dim]
            z_transformer: Transformer embeddings [N, out_dim]

        Returns:
            z: Fused embeddings [N, out_dim]
        """
        if z_gnn.shape != z_transformer.shape:
            raise RuntimeError(
                f"Shape mismatch: GNN {tuple(z_gnn.shape)} vs "
                f"Transformer {tuple(z_transformer.shape)}"
            )

        # Concatenate both embeddings
        combined = torch.cat([z_gnn, z_transformer], dim=-1)

        # Learn gating weights
        gate = self.gate(combined)

        # Weighted combination
        z = gate * z_gnn + (1.0 - gate) * z_transformer

        return z

class GatedFusion(nn.Module):
    """
    Gated fusion with normalization (stable variant).

    - Prevents gate saturation
    - Allows node-wise branch weighting
    """

    def __init__(self, out_dim: int):
        super().__init__()
        self.norm_gnn = nn.LayerNorm(out_dim)
        self.norm_tr  = nn.LayerNorm(out_dim)

        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.Sigmoid()
        )

    def forward(self, z_gnn: torch.Tensor, z_transformer: torch.Tensor) -> torch.Tensor:
        if z_gnn.shape != z_transformer.shape:
            raise RuntimeError(
                f"Shape mismatch: GNN {tuple(z_gnn.shape)} vs "
                f"Transformer {tuple(z_transformer.shape)}"
            )

        z_gnn = self.norm_gnn(z_gnn)
        z_tr  = self.norm_tr(z_transformer)

        gate = self.gate(torch.cat([z_gnn, z_tr], dim=-1))
        return gate * z_gnn + (1.0 - gate) * z_tr

class ClassifierHead(nn.Module):
    def __init__(self, out_dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(out_dim)

        if hidden_dim is not None:
            self.head = nn.Sequential(
                nn.Linear(out_dim, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.head = nn.Linear(out_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.norm(z)
        return self.head(z).view(-1)

Anomaly Head

class EnergyHead(nn.Module):
    """Energy-based anomaly head for fraud detection.

    Higher energy scores indicate more anomalous (fraudulent) nodes.

    Args:
        out_dim: Dimension of input embeddings
        hidden_dim: Optional hidden layer dimension. If None, uses single linear layer
        dropout: Dropout probability for hidden layer
    """

    def __init__(
        self,
        out_dim: int,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dim is not None:
            # Two-layer MLP
            self.head = nn.Sequential(
                nn.Linear(out_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            # Single linear layer
            self.head = nn.Linear(out_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute energy scores for anomaly detection.

        Args:
            z: Node embeddings [N, out_dim]

        Returns:
            energy: Energy scores [N] (higher = more anomalous/fraudulent)
        """
        energy = self.head(z).view(-1)  # [N, 1] -> [N]
        return energy

Training Validation and Testing

def find_best_threshold(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    num_thresholds: int = 100,
) -> Tuple[float, float]:
    """Find threshold that maximizes F1 score on validation set.

    Args:
        logits: Model logits [N]
        y_true: True labels [N]
        num_thresholds: Number of thresholds to try

    Returns:
        best_threshold: Threshold that maximizes F1
        best_f1: Best F1 score achieved
    """
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = y_true.cpu().numpy()

    thresholds = np.linspace(0, 1, num_thresholds)
    best_f1 = 0.0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (probs >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def compute_metrics(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        logits: Model logits [N]
        y_true: True labels [N]
        threshold: Decision threshold for binary prediction

    Returns:
        metrics: Dictionary of metric names to values
    """
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    y_pred = (probs >= threshold).astype(int)

    # Classification metrics
    acc = accuracy_score(y_true_np, y_pred)
    prec = precision_score(y_true_np, y_pred, zero_division=0)
    rec = recall_score(y_true_np, y_pred, zero_division=0)
    f1 = f1_score(y_true_np, y_pred, zero_division=0)

    # Confusion matrix for specificity
    tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Ranking metrics
    roc_auc = roc_auc_score(y_true_np, probs)
    pr_auc = average_precision_score(y_true_np, probs)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'specificity': spec,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': (tn, fp, fn, tp),
    }


def train_epoch(
    model: nn.Module,
    data,
    target_ntype: str,
    optimizer: torch.optim.Optimizer,
    pos_weight: torch.Tensor,
    epoch_num: int,
    *,
    lambda_energy: float = 0.01,
    energy_warmup: int = 10,
    margin: float = 1.0,
    grad_clip: float = 1.0,
    enableAnomaly: bool
) -> Dict[str, float]:

    if enableAnomaly:
      model.train()
      optimizer.zero_grad()

      
      # Forward
      
      output = model(data, target_ntype)

      logits = output.logits
      z = output.z
      energy_head = model.energy_head

      train_mask = data[target_ntype].train_mask
      y = data[target_ntype].y.float()

      logits_tr = logits[train_mask].clamp(-10, 10)
      y_tr = y[train_mask]

      
      # Supervised loss
      
      if epoch_num < 5:
          criterion = nn.BCEWithLogitsLoss()
      else:
          criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

      loss_sup = criterion(logits_tr, y_tr)

      
      # Energy loss (WARMED UP + DETACHED)
      
      loss_energy = logits_tr.new_tensor(0.0)

      if epoch_num >= energy_warmup and lambda_energy > 0:
          with torch.no_grad():
              z_det = z.detach()

          energy = energy_head(z_det).view(-1)
          E = energy[train_mask]
          Y = y_tr

          mask_pos = (Y == 1)
          mask_neg = (Y == 0)

          if mask_pos.any() and mask_neg.any():
              E = (E - E.mean()) / (E.std() + 1e-6)

              E_pos = E[mask_pos]
              E_neg = E[mask_neg]

              n = min(E_pos.numel(), E_neg.numel())
              idx_pos = torch.randperm(E_pos.numel(), device=E.device)[:n]
              idx_neg = torch.randperm(E_neg.numel(), device=E.device)[:n]

              loss_energy = torch.relu(
                  margin - (E_pos[idx_pos] - E_neg[idx_neg])
              ).mean()


              loss_energy = lambda_energy * loss_energy

      
      # Total loss
      
      var_loss = torch.var(logits_tr)
      loss = loss_sup + loss_energy - 0.01 * var_loss
      loss.backward()

      # gradient clipping
      torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

      optimizer.step()

      return {
          "loss": float(loss.item()),
          "loss_sup": float(loss_sup.item()),
          "loss_energy": float(loss_energy.item()),
          "logits_mean": float(logits_tr.mean().item()),
          "logits_std": float(logits_tr.std().item()),
      }

    # If Anomaly is not enabled
    else:

      model.train()
      optimizer.zero_grad()

      
      # Forward
      
      output = model(data, target_ntype)
      logits = output.logits

      train_mask = data[target_ntype].train_mask
      y = data[target_ntype].y.float()

      logits_tr = logits[train_mask].clamp(-10, 10)
      y_tr = y[train_mask]

      
      # Supervised loss
      
      if epoch_num < 5:
          criterion = nn.BCEWithLogitsLoss()
      else:
          criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

      loss_sup = criterion(logits_tr, y_tr)
      var_loss = torch.var(logits_tr)

      # Total loss (pure supervised)
      loss = loss_sup - 0.01 * var_loss


      # Backward

      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
      optimizer.step()

      return {
          "loss": float(loss.item()),
          "loss_sup": float(loss_sup.item()),
          "logits_mean": float(logits_tr.mean().item()),
          "logits_std": float(logits_tr.std().item()),
      }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data,
    target_ntype: str,
    mask_name: str,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Evaluate model on a given split.

    Args:
        model: The fraud detection model
        data: HeteroData object
        target_ntype: Target node type
        mask_name: Name of mask ('train_mask', 'val_mask', or 'test_mask')
        threshold: Decision threshold (if None, use 0.5)

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()

    # Forward pass
    output = model(data, target_ntype)
    logits = output.logits

    # Get mask and labels
    mask = data[target_ntype][mask_name]
    y = data[target_ntype].y

    # Compute metrics
    if threshold is None:
        threshold = 0.5

    metrics = compute_metrics(logits[mask], y[mask], threshold)

    # Also compute loss
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits[mask], y[mask].float()).item()
    metrics['loss'] = loss

    if mask_name == 'test_mask':
      save_path = "/content/drive/MyDrive/Dataset/IEEE-CIS-Prediction.csv"

      probs = torch.sigmoid(logits[mask]).detach().cpu().numpy()
      y_true = y[mask].detach().cpu().numpy().astype(int)
      y_pred = (probs >= threshold).astype(int)

      # --- MODIFICATION START ---
      # Create a new DataFrame for the predictions, rather than loading an existing one
      # This ensures the DataFrame always has the correct number of rows.
      df_predictions = pd.DataFrame({
          "actual": y_true,
          "encoder_pred": y_pred,
          "encoder_prob": probs
      })
      df_predictions.to_csv(save_path, index=False) # Save the predictions to CSV

      # Return metrics and the list of prediction arrays as expected by the caller
      return metrics, [probs, y_true, y_pred]

    else:
      return metrics # Return metrics and loss

def train_validate_test(
    model: nn.Module,
    data,
    target_ntype: str,
    optimizer: torch.optim.Optimizer,
    scheduler,
    num_epochs: int,
    pos_weight: torch.Tensor,
    lambda_energy: float = 0.2,
    margin: float = 1.0,
    patience: int = 20,
    device: str = 'cuda',
    verbose: bool = True,
    enableAnomaly = bool
) -> Dict:
    """Full training loop with early stopping and threshold selection.

    Args:
        model: The fraud detection model
        data: HeteroData object
        target_ntype: Target node type
        optimizer: Optimizer
        scheduler: Learning rate scheduler (ReduceLROnPlateau)
        num_epochs: Maximum number of epochs
        pos_weight: Positive class weight
        lambda_energy: Energy loss weight
        margin: Energy margin
        patience: Early stopping patience
        device: Device to use
        verbose: Whether to print detailed logs

    Returns:
        results: Dictionary with training history and best model
    """
    best_val_pr_auc = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = []

    if verbose:
        print(f"\n{'='*80}")
        print(f"{'TRAINING STARTED':^80}")
        print(f"{'='*80}")
        print(f"Configuration")

    for epoch in range(num_epochs):
        # Train

        if enableAnomaly:
          train_metrics = train_epoch(
              model=model,
              data=data,
              target_ntype=target_ntype,
              optimizer=optimizer,
              pos_weight=pos_weight,
              epoch_num=epoch,
              lambda_energy=lambda_energy,
              margin=margin,
              enableAnomaly = enableAnomaly
          )
        else:
          train_metrics = train_epoch(
              model=model,
              data=data,
              target_ntype=target_ntype,
              optimizer=optimizer,
              pos_weight=pos_weight,
              epoch_num=epoch,
              lambda_energy=None,
              margin=None,
              enableAnomaly = enableAnomaly
          )

        # Evaluate on train and val
        train_eval = evaluate(model, data, target_ntype, 'train_mask')
        val_eval = evaluate(model, data, target_ntype, 'val_mask')

        # Find best threshold on validation set
        with torch.no_grad():
            model.eval()
            output = model(data, target_ntype)
            val_mask = data[target_ntype].val_mask
            best_threshold, _ = find_best_threshold(
                output.logits[val_mask],
                data[target_ntype].y[val_mask]
            )

        # Step scheduler
        scheduler.step(val_eval['pr_auc'])

        # Early stopping check
        if val_eval['pr_auc'] > best_val_pr_auc:
            best_val_pr_auc = val_eval['pr_auc']
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        # Log progress
        history.append({
            'epoch': epoch,
            'train': train_eval,
            'val': val_eval,
            'loss': train_metrics,
            'threshold': best_threshold,
        })

        # Print every epoch
        if verbose:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Loss: {train_metrics['loss']:.4f} "
                  f"(Sup: {train_metrics['loss_sup']:.4f}, "
                  f"Energy: {train_metrics['loss_energy']:.4f}) | "
                  f"Train - Loss: {train_eval['loss']:.4f}, ROC: {train_eval['roc_auc']:.4f}, PR: {train_eval['pr_auc']:.4f} | "
                  f"Val - Loss: {val_eval['loss']:.4f}, ROC: {val_eval['roc_auc']:.4f}, PR: {val_eval['pr_auc']:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Print best model update
            if val_eval['pr_auc'] > best_val_pr_auc:
                print(f"  → New best model! Val PR-AUC improved: {best_val_pr_auc:.4f} → {val_eval['pr_auc']:.4f}")

        # Early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"\n{'='*80}")
                print(f"Early stopping triggered at epoch {epoch+1}")
                print(f"Best epoch was {best_epoch+1} with Val PR-AUC: {best_val_pr_auc:.4f}")
                print(f"{'='*80}\n")
            break

    # Load best model
    model.load_state_dict(best_state)

    # Final evaluation on test set with best threshold
    with torch.no_grad():
        model.eval()
        output = model(data, target_ntype)
        val_mask = data[target_ntype].val_mask
        best_threshold, _ = find_best_threshold(
            output.logits[val_mask],
            data[target_ntype].y[val_mask]
        )

    test_metrics, df = evaluate(model, data, target_ntype, 'test_mask', best_threshold)

    if verbose:
        print(f"\n{'='*80}")
        print(f"{'TRAINING COMPLETED':^80}")
        print(f"{'='*80}")
        print(f"Best Model: Epoch {best_epoch+1} | Best Threshold: {best_threshold:.4f}")
        print(f"{'-'*80}")
        print(f"{'FINAL TEST RESULTS':^80}")
        print(f"{'-'*80}")
        print(f"Loss:        {test_metrics['loss']:.4f}")
        print(f"ROC-AUC:     {test_metrics['roc_auc']:.4f}")
        print(f"PR-AUC:      {test_metrics['pr_auc']:.4f}")
        print(f"Accuracy:    {test_metrics['accuracy']:.4f}")
        print(f"Precision:   {test_metrics['precision']:.4f}")
        print(f"Recall:      {test_metrics['recall']:.4f}")
        print(f"F1 Score:    {test_metrics['f1']:.4f}")
        print(f"Specificity: {test_metrics['specificity']:.4f}")
        tn, fp, fn, tp = test_metrics['confusion_matrix']
        print(f"{'-'*80}")
        print(f"Confusion Matrix:")
        print(f"  True Negatives:  {tn:6d}  |  False Positives: {fp:6d}")
        print(f"  False Negatives: {fn:6d}  |  True Positives:  {tp:6d}")
        print(f"{'='*80}\n")

    return {
        'history': history,
        'best_epoch': best_epoch,
        'best_threshold': best_threshold,
        'test_metrics': test_metrics,
        'dataframe': df
    }

Main model Pipeline

@dataclass
class ModelOutput:
    logits: torch.Tensor
    energy: torch.Tensor
    z: torch.Tensor

class PureHeteroFraudDetector(nn.Module):
    """Pure heterogeneous fraud detection model."""

    def __init__(
        self,
        in_dims: Dict[str, int],
        hidden_dim: int,
        out_dim: int,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        alpha: float = 0.5,
        num_gnn_layers: int = 2,
        num_hgt_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.node_types = metadata[0]
        self.metadata = metadata
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # 1. Per-type projection
        self.projector = HeteroProjector(
            in_dims=in_dims,
            hidden_dim=hidden_dim,
            node_types=self.node_types,
            dropout = dropout
        )

        # 2. Hetero GNN encoder
        self.gnn_encoder = HeteroSAGEEncoder(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            metadata=metadata,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )

        # 3. Hetero Transformer encoder
        self.transformer_encoder = HGTEncoder(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            metadata=metadata,
            num_heads=num_heads,
            num_layers=num_hgt_layers,
            dropout=dropout,
        )

        # Branch normalization (pre-fusion)
        self.norm_gnn = nn.LayerNorm(out_dim)
        self.norm_transformer = nn.LayerNorm(out_dim)

        # 4. Fusion
        self.fusion = GatedFusion(out_dim=out_dim)

        # Pre-classifier normalization
        self.norm_fused = nn.LayerNorm(out_dim)

        # 5. Classifier head
        self.classifier = ClassifierHead(
            out_dim=out_dim,
            hidden_dim=None,
            dropout=dropout,
        )

        # Energy head normalization
        self.norm_energy = nn.LayerNorm(out_dim)

        # 6. Energy head
        self.energy_head = EnergyHead(
            out_dim=out_dim,
            hidden_dim=None,
            dropout=dropout,
        )

    def forward(self, data, target_ntype: str) -> ModelOutput:
        if target_ntype not in self.node_types:
            raise ValueError(
                f"target_ntype='{target_ntype}' not in metadata node types"
            )

        # 1. Project inputs
        x_dict = self.projector(data)

        # 2. GNN branch
        z_gnn_dict = self.gnn_encoder(x_dict, data.edge_index_dict)
        z_gnn = self.norm_gnn(z_gnn_dict[target_ntype])

        # 3. Transformer branch
        z_tr_dict = self.transformer_encoder(x_dict, data.edge_index_dict)
        z_tr = self.norm_transformer(z_tr_dict[target_ntype])

        # 4. Fuse
        z = self.fusion(z_gnn, z_tr)

        # stabilize before classifier
        z_cls = self.norm_fused(z)

        logits = self.classifier(z_cls)
        logits = logits.clamp(min=-20, max=20)

        # # stable energy features
        z_energy = self.norm_energy(z.detach())
        energy = self.energy_head(z_energy)

        return ModelOutput(
            logits=logits,
            energy=energy,
            z = z
        )

# Execution

def run_fraud_detection_training_dynamic(
    data,
    *,
    target_ntype: str = "transaction",
    device: str = "cuda",
    num_epochs: int = 200,
    enableAnomaly=bool,
    seeds=[13, 42, 137, 2023, 9999],
):
    """
    Fully dynamic fraud detection training pipeline (multi-seed enabled).

    Additions (minimal, no pipeline distortion):
      - Collect per-seed probs from results['dataframe'] = [probs, true, preds]
      - Compute/store PR-AUC per seed
      - Return pr_auc_df + avg_probs_df in final_results['multi_seed_summary']
    """

    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    print("=" * 90)
    print("DYNAMIC HETERO FRAUD DETECTION TRAINING".center(90))
    print("=" * 90)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    print(f"Device: {device}")
    print(f"Target node type: {target_ntype}")

    node_types = list(data.node_types)
    edge_types = list(data.edge_types)
    metadata = (node_types, edge_types)

    in_dims = {}
    for ntype in node_types:
        if "x" not in data[ntype]:
            raise RuntimeError(f"Node type '{ntype}' has no features 'x'")
        in_dims[ntype] = data[ntype].x.size(-1)

    print("\nInferred input dimensions:")
    for k, v in in_dims.items():
        print(f"  {k}: {v}")

    y = data[target_ntype].y
    train_mask = data[target_ntype].train_mask

    num_pos = y[train_mask].sum().item()
    num_neg = (y[train_mask] == 0).sum().item()
    pos_weight = torch.tensor([num_neg / max(num_pos, 1.0)], device=device)

    print("\nClass statistics (train):")
    print(f"  Fraud:  {num_pos}")
    print(f"  Normal: {num_neg}")
    print(f"  pos_weight: {pos_weight.item():.2f}")

    all_metrics = []
    final_model = None
    final_results = None

    # NEW: store PR-AUC per seed + probs per seed for averaging
    pr_auc_rows = []
    per_seed_probs = []
    y_ref = None

    
    # Multi-Seed Loop
    
    for seed in seeds:

        print("\n" + "-" * 90)
        print(f"Running Seed: {seed}")
        print("-" * 90)

        set_seed(seed)

        model = PureHeteroFraudDetector(
            in_dims=in_dims,
            hidden_dim=128,
            out_dim=64,
            metadata=metadata,
            alpha=0.65,
            dropout=0.1,
            num_gnn_layers=3,
            num_heads=2
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            weight_decay=2e-4,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=10,
        )

        results = train_validate_test(
            model=model,
            data=data,
            target_ntype=target_ntype,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            pos_weight=pos_weight,
            lambda_energy=0.02,
            margin=1.0,
            patience=20,
            device=device,
            verbose=True,
            enableAnomaly=enableAnomaly
        )

        # ---- metrics per seed (as-is) ----
        test_metrics = dict(results["test_metrics"])
        test_metrics["seed"] = seed

        # NEW: pull [probs, true, preds] from results['dataframe']
        if "dataframe" not in results:
            raise KeyError("Expected results to include key 'dataframe' = [probs, true, preds].")

        listing = results["dataframe"]
        if not (isinstance(listing, (list, tuple)) and len(listing) >= 2):
            raise ValueError("results['dataframe'] must be [probs, true, preds].")

        probs = np.asarray(listing[0], dtype=float).ravel()
        y_true = np.asarray(listing[1], dtype=int).ravel()

        # Ensure consistent labels across seeds if you plan to average probs
        if y_ref is None:
            y_ref = y_true
        else:
            if len(y_true) != len(y_ref) or not np.array_equal(y_true, y_ref):
                raise ValueError(
                    "y_true mismatch across seeds. "
                    "If you want to average probs across seeds, you must ensure test ordering is identical."
                )

        pr_auc = average_precision_score(y_true, probs)
        test_metrics["pr_auc"] = pr_auc

        all_metrics.append(test_metrics)
        pr_auc_rows.append({"seed": seed, "pr_auc": pr_auc})
        per_seed_probs.append(probs)

        # Save last seed model/results
        if seed == seeds[-1]:
            final_model = model
            final_results = results

    
    # Aggregate Summary
    
    metrics_df = pd.DataFrame(all_metrics)

    # PR-AUC per seed dataframe
    pr_auc_df = pd.DataFrame(pr_auc_rows)

    # Average probs across seeds dataframe
    probs_mat = np.vstack(per_seed_probs)  # (n_seeds, n_samples)
    avg_prob = probs_mat.mean(axis=0)
    avg_probs_df = pd.DataFrame({
        "actual": y_ref,
        "avg_prob": avg_prob
    })

    summary = {
        "mean": metrics_df.drop(columns=["seed"], errors="ignore").mean(numeric_only=True).to_dict(),
        "median": metrics_df.drop(columns=["seed"], errors="ignore").median(numeric_only=True).to_dict(),
        "std": metrics_df.drop(columns=["seed"], errors="ignore").std(numeric_only=True).to_dict(),
        "all_runs": metrics_df,
        # NEW requested artifacts:
        "pr_auc_df": pr_auc_df,
        "avg_probs_df": avg_probs_df
    }

    # Inject into last results dictionary
    final_results["multi_seed_summary"] = summary

    print("\n" + "=" * 90)
    print("MULTI-SEED SUMMARY".center(90))
    print("=" * 90)
    print("Mean Metrics:")
    print(summary["mean"])
    print("=" * 90)

    return final_model, final_results
