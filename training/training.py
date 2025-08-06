import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple, Dict, List
import time

from src.utils.config import Config, default_config
from src.utils.utils import setup_logging, timer, validate_tensor, log_memory_usage, set_random_seeds, MetricsTracker, load_train_val_test_masks, validate_masks_no_leakage, filter_labeled_nodes, cache_eigenvectors, setup_pretrained_training
from src.models.models import GraceStyleModel, AnomalyClassifier, nt_xent_loss, create_model, build_augmented_graph, combined_adgcl_loss, DSGADModel, sample_wavelet_band_masks, create_mae_mask, combined_adgcl_mae_loss, TemporalGraphData, TemporalDSGADModel, temporal_consistency_loss, temporal_augmentation, EnergyDistanceClassifier, EnsembleEnergyClassifier, energy_distance_loss


class GADTrainer:
    """Graph Anomaly Detection Trainer with simplified pipeline"""
    
    def __init__(self, config: Optional[Config] = None, use_dsgad: bool = True, 
                 pretrained_path: Optional[str] = None, freeze_backbone: bool = False,
                 use_temporal: bool = False, window_size: int = 5, temporal_overlap: float = 0.5,
                 use_energy_scoring: bool = False, ensemble_energy: bool = False, 
                 energy_metrics: List[str] = ["euclidean", "cosine"]):
        self.config = config or default_config
        self.logger = setup_logging(self.config.system.log_level, self.config.system.log_file)
        self.device = torch.device(self.config.system.device)
        self.use_dsgad = use_dsgad
        self.pretrained_path = pretrained_path
        self.freeze_backbone = freeze_backbone
        self.use_temporal = use_temporal
        self.window_size = window_size
        self.temporal_overlap = temporal_overlap
        self.use_energy_scoring = use_energy_scoring
        self.ensemble_energy = ensemble_energy
        self.energy_metrics = energy_metrics
        
        # random seeds for reproducibility
        set_random_seeds(self.config.system.random_seed, self.logger)
        
        # initialize components
        self.data = None
        self.model = None
        self.classifier = None
        self.metrics_tracker = MetricsTracker(self.logger)
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        self.last_embeddings = None  # neighbor completion
        self.temporal_data = None  # temporal modeling
        self.energy_classifier = None  # energy distance scoring

        if self.config.training.save_checkpoints:
            Path(self.config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def load_data(self, data_path: str = None, use_masks: bool = True):
        """Load preprocessed graph data and train/val/test masks"""
        data_path = data_path or self.config.data.static_save_path
        
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        with timer(self.logger, "data loading"):
            self.data = torch.load(data_path, weights_only=False, map_location='cpu').to(self.device)
            self.config.model.in_channels = self.data.x.size(1)
            self.logger.info(f"Loaded graph: {self.data.num_nodes} nodes, {self.data.num_edges} edges")
            self.logger.info(f"Feature dimensions: {self.data.x.size(1)}")
            self.logger.info(f"Label distribution: {torch.bincount(self.data.y[self.data.y >= 0])}")
            
            # load train/val/test masks if requested
            if use_masks:
                self.load_masks()
            
            # cache eigenvectors for spectral methods
            if self.use_dsgad or self.use_temporal:
                with timer(self.logger, "eigenvector computation"):
                    cache_eigenvectors(self.data, k=64, logger=self.logger)
            
            # initialize temporal data structure if temporal modeling is enabled
            if self.use_temporal:
                self.temporal_data = TemporalGraphData(
                    window_size=self.window_size,
                    overlap=self.temporal_overlap
                )
                # add current snapshot as initial temporal state
                self.temporal_data.add_snapshot(0, self.data.clone())
                self.temporal_data.compute_temporal_edges()
                self.logger.info(f"Initialized temporal modeling with window_size={self.window_size}")
    
    def load_masks(self):
        """Load train/validation/test masks and validate no data leakage"""
        try:
            data_dir = Path(self.config.data.features_path).parent

            self.train_mask, self.val_mask, self.test_mask = load_train_val_test_masks(
                str(data_dir), self.logger
            )

            # ensure no data leakage
            if not validate_masks_no_leakage(self.train_mask, self.val_mask, self.test_mask, self.logger):
                raise ValueError("Data leakage detected between train/val/test masks!")
            
            # filter to only labeled nodes
            self.train_mask = filter_labeled_nodes(self.train_mask, self.data.y)
            self.val_mask = filter_labeled_nodes(self.val_mask, self.data.y)
            self.test_mask = filter_labeled_nodes(self.test_mask, self.data.y)
            
            self.logger.info(f"Filtered masks - Train: {len(self.train_mask)}, Val: {len(self.val_mask)}, Test: {len(self.test_mask)}")
            
        except Exception as e:
            self.logger.error(f"Failed to load masks: {str(e)}")
            self.logger.warning("Falling back to using all labeled data (DATA LEAKAGE!)")
            labeled_mask = (self.data.y == 0) | (self.data.y == 1)
            labeled_nodes = torch.where(labeled_mask)[0]
            n_nodes = len(labeled_nodes)
            
            # 60% train, 20% val, 20% test
            train_end = int(0.6 * n_nodes)
            val_end = int(0.8 * n_nodes)
            
            self.train_mask = labeled_nodes[:train_end]
            self.val_mask = labeled_nodes[train_end:val_end]
            self.test_mask = labeled_nodes[val_end:]
    
    def setup_models(self):
        """Initialize models"""
        if self.data is None:
            raise ValueError("Data must be loaded before setting up models")
        
        # create models based on architecture choice
        if self.use_temporal:
            self.logger.info("Using Temporal DSGAD (temporal + spectral + structural) model")
            self.model = TemporalDSGADModel(
                in_channels=self.data.x.size(1),
                hidden_channels=self.config.model.hidden_channels,
                out_channels=self.config.model.latent_dim,
                proj_dim=self.config.model.projection_dim,
                dropout=self.config.model.dropout,
                num_spectral_bands=4,
                window_size=self.window_size
            ).to(self.device)
        elif self.use_dsgad:
            self.logger.info("Using DSGAD (spectral + structural) model")
            self.model = DSGADModel(
                in_channels=self.data.x.size(1),
                hidden_channels=self.config.model.hidden_channels,
                out_channels=self.config.model.latent_dim,
                proj_dim=self.config.model.projection_dim,
                dropout=self.config.model.dropout,
                num_spectral_bands=4
            ).to(self.device)
        else:
            self.logger.info("Using GRACE-style (structural only) model")
            self.model = GraceStyleModel(
                in_channels=self.data.x.size(1),
                hidden_channels=self.config.model.hidden_channels,
                out_channels=self.config.model.latent_dim,
                proj_dim=self.config.model.projection_dim,
                dropout=self.config.model.dropout
            ).to(self.device)
        
        self.classifier = AnomalyClassifier(
            in_dim=self.config.model.latent_dim,
            hidden_dim=self.config.model.classifier_hidden,
            dropout=self.config.model.dropout
        ).to(self.device)
        
        # energy distance classifier if requested
        if self.use_energy_scoring:
            if self.ensemble_energy:
                # ensemble energy classifier for multiple embedding views
                embedding_dims = [self.config.model.latent_dim]  # base embedding
                if self.use_dsgad or self.use_temporal:
                    embedding_dims.append(self.config.model.latent_dim)  # spectral view
                if self.use_temporal:
                    embedding_dims.append(self.config.model.latent_dim)  # temporal view
                
                self.energy_classifier = EnsembleEnergyClassifier(
                    embedding_dims=embedding_dims,
                    ensemble_method="weighted_voting",
                    num_prototypes=10
                ).to(self.device)
            else:
                # single energy classifier
                self.energy_classifier = EnergyDistanceClassifier(
                    embedding_dim=self.config.model.latent_dim,
                    num_prototypes=10,
                    distance_metrics=self.energy_metrics,
                    adaptive_threshold=True,
                    ensemble_weights=True
                ).to(self.device)
        
        # pre-trained initialization if requested
        if self.pretrained_path:
            self.pretrained_config = setup_pretrained_training(
                self.model, self.pretrained_path, 
                freeze_backbone=self.freeze_backbone,
                backbone_lr=1e-5,  # Phase 4 spec
                logger=self.logger
            )
        else:
            self.pretrained_config = None
        
        self.logger.info(f"Models initialized successfully (DSGAD: {self.use_dsgad}, Temporal: {self.use_temporal}, Energy: {self.use_energy_scoring}, Pre-trained: {self.pretrained_path is not None})")
    
    def create_data_loaders(self) -> Tuple[NeighborLoader, NeighborLoader]:
        """Create data loaders for training and evaluation"""
        pretrain_loader = NeighborLoader(
            self.data,
            input_nodes=None,
            num_neighbors=self.config.training.num_neighbors,
            batch_size=self.config.training.pretrain_batch_size,
            shuffle=True,
            num_workers=self.config.system.num_workers
        )
        
        eval_loader = NeighborLoader(
            self.data,
            input_nodes=None,
            num_neighbors=[-1],  # all neighbors for evaluation
            batch_size=self.config.training.eval_batch_size,
            shuffle=False,
            num_workers=self.config.system.num_workers
        )
        
        return pretrain_loader, eval_loader
    
    def simulate_temporal_snapshots(self, num_snapshots: int = 10):
        """
        Simulate temporal snapshots for temporal modeling
        Creates multiple graph snapshots with slight variations to simulate temporal evolution 
        """
        if not self.use_temporal or self.temporal_data is None:
            return
        
        self.logger.info(f"Simulating {num_snapshots} temporal snapshots")
        
        for t in range(1, num_snapshots + 1):
            # copy of the original graph
            snapshot = self.data.clone()
            
            # add temporal variations
            noise = torch.randn_like(snapshot.x) * 0.02
            snapshot.x = snapshot.x + noise
            
            # randomly drop edges
            if snapshot.edge_index.size(1) > 0:
                edge_mask = torch.rand(snapshot.edge_index.size(1)) > 0.05  # Drop 5% of edges
                snapshot.edge_index = snapshot.edge_index[:, edge_mask]
            
            # add some new random edges 
            num_nodes = snapshot.num_nodes
            num_new_edges = max(1, int(snapshot.edge_index.size(1) * 0.02))  # Add 2% new edges
            
            new_src = torch.randint(0, num_nodes, (num_new_edges,))
            new_dst = torch.randint(0, num_nodes, (num_new_edges,))
            new_edges = torch.stack([new_src, new_dst], dim=0)

            snapshot.edge_index = torch.cat([snapshot.edge_index, new_edges], dim=1)
            self.temporal_data.add_snapshot(t, snapshot)
        
        # temporal edges
        self.temporal_data.compute_temporal_edges()
        self.logger.info(f"Temporal snapshots created: {len(self.temporal_data.snapshots)} total snapshots")
    
    def augment_batch_advanced(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[list], Optional[list]]:
        """Create augmented views using AD-GCL neighbor-pruning and neighbor-completion + spectral masking"""
        
        # create original embeddings if available
        if self.last_embeddings is not None:
            # use subset of embeddings 
            batch_embeddings = self.last_embeddings[batch.n_id[:batch.batch_size]] if hasattr(batch, 'n_id') else self.last_embeddings
        else:
            batch_embeddings = None
        
        # augmented views using advanced techniques
        # view 1: neighbor pruning + feature masking
        x1, ei1 = build_augmented_graph(batch, mode="prune", 
                                       last_embeddings=batch_embeddings,
                                       prune_ratio=0.4)
        from models import random_feature_mask
        x1 = random_feature_mask(x1, self.config.training.feature_mask_prob)
        
        # view 2:neighbor completion + feature masking  
        x2, ei2 = build_augmented_graph(batch, mode="complete",
                                       last_embeddings=batch_embeddings,
                                       k_neighbors=3)
        x2 = random_feature_mask(x2, self.config.training.feature_mask_prob)
        
        # original view for inter-view consistency
        x_orig = batch.x
        ei_orig = batch.edge_index
        
        # create spectral masks for DSGAD (if using spectral model)
        spectral_masks1 = None
        spectral_masks2 = None
        if self.use_dsgad and hasattr(self.data, 'U'):
            #sample different wavelet band masks for each view
            U_batch = self.data.U[batch.n_id[:batch.batch_size]] if hasattr(batch, 'n_id') else self.data.U
            spectral_masks1 = sample_wavelet_band_masks(U_batch, num_bands=4, mask_prob=0.3)
            spectral_masks2 = sample_wavelet_band_masks(U_batch, num_bands=4, mask_prob=0.3)
        
        return x1, x2, ei1, ei2, x_orig, ei_orig, spectral_masks1, spectral_masks2
    
    def augment_batch_simple(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create two simple augmented views (fallback)"""
        from models import random_edge_dropout, random_feature_mask
        
        ei1 = random_edge_dropout(batch.x, batch.edge_index, self.config.training.edge_dropout_prob)[1]
        ei2 = random_edge_dropout(batch.x, batch.edge_index, self.config.training.edge_dropout_prob)[1]
        x1 = random_feature_mask(batch.x, self.config.training.feature_mask_prob)
        x2 = random_feature_mask(batch.x, self.config.training.feature_mask_prob)
        
        return x1, x2, ei1, ei2
    
    def pretrain_contrastive(self):
        """Contrastive pretraining phase"""
        self.logger.info("Starting contrastive pretraining")
        
        # for temporal models, simulate temporal snapshots first
        if self.use_temporal:
            self.simulate_temporal_snapshots(num_snapshots=15)
        
        pretrain_loader, _ = self.create_data_loaders()
        
        # use pre-trained config is available
        if self.pretrained_config:
            optimizer = torch.optim.Adam(
                self.pretrained_config['param_groups'],
                weight_decay=self.config.training.weight_decay
            )
            self.logger.info(f"Using pre-trained optimizer (frozen: {self.pretrained_config['frozen_backbone']})")
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
            self.logger.info("Using standard optimizer")
        
        self.model.train()
        
        # lambda values for consistency losses 
        lambda_intra = 0.1
        lambda_inter = 0.1
        lambda_recon = 1.0  
        lambda_temporal = 0.1  
        
        for epoch in range(1, self.config.training.epochs_pretrain + 1):
            epoch_losses = {'total': 0.0, 'contrastive': 0.0, 'intra': 0.0, 'inter': 0.0, 'reconstruction': 0.0, 'temporal': 0.0}
            
            for batch in pretrain_loader:
                batch = batch.to(self.device)
                
                # try advanced augmentations first, fallback to simple if needed
                try:
                    x1, x2, ei1, ei2, x_orig, ei_orig, spec_masks1, spec_masks2 = self.augment_batch_advanced(batch)
                    use_advanced = True
                except Exception as e:
                    self.logger.warning(f"Advanced augmentation failed, using simple: {e}")
                    x1, x2, ei1, ei2 = self.augment_batch_simple(batch)
                    spec_masks1 = spec_masks2 = None
                    use_advanced = False
                
                # MAE mask for reconstruction 
                x_masked, mae_mask = create_mae_mask(batch.x, mask_ratio=0.3)
                
                # forward pass on augmented views
                if self.use_temporal:
                    U_batch = self.data.U[batch.n_id[:batch.batch_size]] if hasattr(batch, 'n_id') else self.data.U
                    current_time = epoch % len(self.temporal_data.snapshots)
                    temporal_aug1 = temporal_augmentation(self.temporal_data, ["dropout", "noise"])
                    temporal_aug2 = temporal_augmentation(self.temporal_data, ["shuffle", "noise"]) 
                    z1, p1 = self.model(temporal_aug1, current_time, U_batch, spec_masks1)
                    z2, p2 = self.model(temporal_aug2, current_time, U_batch, spec_masks2)
                    x_recon = None  # Temporal model doesn't support MAE yet
                    
                    if use_advanced:
                        z_orig, _ = self.model(self.temporal_data, current_time, U_batch)
                
                elif self.use_dsgad:
                    U_batch = self.data.U[batch.n_id[:batch.batch_size]] if hasattr(batch, 'n_id') else self.data.U
                    z1, p1 = self.model(x1, ei1, U_batch, spec_masks1)
                    z2, p2 = self.model(x2, ei2, U_batch, spec_masks2)
                    
                    # MAE reconstruction on masked features
                    _, _, x_recon = self.model(x_masked, batch.edge_index, U_batch, return_reconstruction=True)
                    
                    if use_advanced:
                        z_orig, _ = self.model(x_orig, ei_orig, U_batch)
                else:
                    # GRACE model (structural only) 
                    z1, p1 = self.model(x1, ei1)
                    z2, p2 = self.model(x2, ei2)
                    x_recon = None
                    
                    if use_advanced:
                        z_orig, _ = self.model(x_orig, ei_orig)
                
                if use_advanced:
                    # use average of augmented views as z_aug
                    z_aug = (z1 + z2) / 2
                    
                    # combined loss (AD-GCL + MAE + temporal)
                    if self.use_temporal:
                        base_loss, base_dict = combined_adgcl_loss(
                            p1, p2, z1, z2, z_orig, z_aug,
                            temperature=self.config.training.temperature,
                            lambda_intra=lambda_intra,
                            lambda_inter=lambda_inter
                        )
                        
                        # temporal consistency loss
                        temporal_loss = 0.0
                        if len(self.temporal_data.snapshots) >= 3:
                            # embeddings from consecutive snapshots for temporal consistency
                            prev_time = max(0, current_time - 1)
                            next_time = min(len(self.temporal_data.snapshots) - 1, current_time + 1)
                            
                            if prev_time != current_time and next_time != current_time:
                                z_prev, _ = self.model(self.temporal_data, prev_time, U_batch)
                                z_next, _ = self.model(self.temporal_data, next_time, U_batch)
                                temporal_loss = temporal_consistency_loss(z_prev, z_orig, z_next, lambda_temporal)
                        
                        loss = base_loss + temporal_loss
                        loss_dict = base_dict.copy()
                        loss_dict['temporal'] = temporal_loss.item() if isinstance(temporal_loss, torch.Tensor) else temporal_loss
                        loss_dict['total'] = loss.item()
                        
                    elif self.use_dsgad and x_recon is not None:
                        # loss with MAE reconstruction
                        loss, loss_dict = combined_adgcl_mae_loss(
                            p1, p2, z1, z2, z_orig, z_aug,
                            batch.x, x_recon, mae_mask,
                            temperature=self.config.training.temperature,
                            lambda_intra=lambda_intra,
                            lambda_inter=lambda_inter,
                            lambda_recon=lambda_recon
                        )
                    else:
                        # AD-GCL only
                        loss, loss_dict = combined_adgcl_loss(
                            p1, p2, z1, z2, z_orig, z_aug,
                            temperature=self.config.training.temperature,
                            lambda_intra=lambda_intra,
                            lambda_inter=lambda_inter
                        )
                    
                    # individual loss components
                    for key, value in loss_dict.items():
                        if key in epoch_losses:
                            epoch_losses[key] += value
                else:
                    # simple contrastive loss
                    loss = nt_xent_loss(p1, p2, self.config.training.temperature)
                    epoch_losses['contrastive'] += loss.item()
                    epoch_losses['total'] += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.gradient_clip_norm
                )
                optimizer.step()
            
            # avg losses over batches
            for key in epoch_losses:
                epoch_losses[key] /= len(pretrain_loader)
            
            self.metrics_tracker.update(**epoch_losses)
            self.metrics_tracker.log_epoch(epoch, "pretrain")
            
            if epoch % 5 == 0:  
                with torch.no_grad():
                    self.last_embeddings = self.embed_full_graph()
            
            if (self.config.training.save_checkpoints and 
                epoch % self.config.training.save_every_n_epochs == 0):
                self.save_checkpoint(epoch, "pretrain")
        
        self.logger.info("Contrastive pretraining completed")
    
    def embed_full_graph(self) -> torch.Tensor:
        """Generate embeddings for the full graph"""
        self.model.eval()
        embeddings = []
        
        _, eval_loader = self.create_data_loaders()
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = batch.to(self.device)
                
                if self.use_temporal:
                    U_batch = self.data.U[batch.n_id[:batch.batch_size]] if hasattr(batch, 'n_id') else self.data.U
                    current_time = len(self.temporal_data.snapshots) - 1  # Use latest snapshot
                    z_batch = self.model.encode(self.temporal_data, current_time, U_batch)
                elif self.use_dsgad:
                    # requires eigenvectors
                    U_batch = self.data.U[batch.n_id[:batch.batch_size]] if hasattr(batch, 'n_id') else self.data.U
                    z_batch = self.model.encode(batch.x, batch.edge_index, U_batch)
                else:
                    # GRACE 
                    z_batch = self.model.encode(batch.x, batch.edge_index)
                
                # only take embeddings for seed nodes
                embeddings.append(z_batch[:batch.batch_size].cpu())
        
        return torch.cat(embeddings, dim=0)
    
    def extract_multiple_embeddings(self) -> List[torch.Tensor]:
        """
        Extract multiple embedding views for ensemble energy scoring
        
        Returns:
            List of embeddings: [spatial, spectral, temporal] depending on model configuration
        """
        self.model.eval()
        embedding_views = []
        
        _, eval_loader = self.create_data_loaders()
        
        with torch.no_grad():
            spatial_embeddings = []
            spectral_embeddings = []
            temporal_embeddings = []
            
            for batch in eval_loader:
                batch = batch.to(self.device)
                
                if self.use_temporal:
                    U_batch = self.data.U[batch.n_id[:batch.batch_size]] if hasattr(batch, 'n_id') else self.data.U
                    current_time = len(self.temporal_data.snapshots) - 1
                    
                    # embeddings from temporal model's encoder
                    encoder = self.model.encoder
                    
                    # spatial-temporal pathway
                    z_spatial_temporal = encoder.temporal_gat2(
                        encoder.temporal_gat1(batch.x, batch.edge_index), 
                        batch.edge_index
                    )
                    
                    # spectral pathway  
                    z_spectral = encoder.spectral2(
                        encoder.spectral1(batch.x, U_batch), 
                        U_batch
                    )
                    
                    # temporal aggregation pathway (simplified)
                    z_temporal = encoder.temporal_fusion(
                        torch.cat([z_spectral] * encoder.window_size, dim=1)
                    )
                    
                    spatial_embeddings.append(z_spatial_temporal[:batch.batch_size].cpu())
                    spectral_embeddings.append(z_spectral[:batch.batch_size].cpu())
                    temporal_embeddings.append(z_temporal[:batch.batch_size].cpu())
                    
                elif self.use_dsgad:
                    # DSGAD - extract spatial and spectral views
                    U_batch = self.data.U[batch.n_id[:batch.batch_size]] if hasattr(batch, 'n_id') else self.data.U
                    
                    # spatial pathway (GAT)
                    z_spatial = self.model.encoder.gat2(
                        F.elu(self.model.encoder.gat1(batch.x, batch.edge_index)), 
                        batch.edge_index
                    )
                    
                    # spectral pathway
                    z_spectral = self.model.encoder.spectral2(
                        self.model.encoder.spectral1(batch.x, U_batch), 
                        U_batch
                    )
                    
                    spatial_embeddings.append(z_spatial[:batch.batch_size].cpu())
                    spectral_embeddings.append(z_spectral[:batch.batch_size].cpu())
                    
                else:
                    # GRACE model - single spatial embedding
                    z_spatial = self.model.encode(batch.x, batch.edge_index)
                    spatial_embeddings.append(z_spatial[:batch.batch_size].cpu())
        
        # concatenate all embeddings for each view
        embedding_views.append(torch.cat(spatial_embeddings, dim=0))
        
        if spectral_embeddings:
            embedding_views.append(torch.cat(spectral_embeddings, dim=0))
            
        if temporal_embeddings:
            embedding_views.append(torch.cat(temporal_embeddings, dim=0))
        
        return embedding_views
    
    def finetune_classifier(self, early_stopping: bool = True):
        """Fine-tune classifier on training data with validation-based early stopping"""
        self.logger.info("Starting classifier fine-tuning...")
        
        if self.train_mask is None:
            raise ValueError("Training masks not loaded. Call load_data() first.")
        
        # generate embeddings
        with timer(self.logger, "embedding generation"):
            if self.use_energy_scoring and self.ensemble_energy:
                embedding_views = self.extract_multiple_embeddings()
                embeddings = embedding_views[0]  # Use spatial embeddings for standard classifier
            else:
                embeddings = self.embed_full_graph()
                embedding_views = [embeddings] if self.use_energy_scoring else None
        
        # create training dataset using only train_mask
        train_features = embeddings[self.train_mask.cpu()]
        train_labels = self.data.y[self.train_mask].float().cpu()
        
        train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.config.training.finetune_batch_size,
            shuffle=True
        )
        
        # create validation dataset for early stopping
        if early_stopping and len(self.val_mask) > 0:
            val_features = embeddings[self.val_mask.cpu()]
            val_labels = self.data.y[self.val_mask].float().cpu()
            val_dataset = torch.utils.data.TensorDataset(val_features, val_labels)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=self.config.training.eval_batch_size,
                shuffle=False
            )
        else:
            val_loader = None
        
        # setup optimizer - use separate learning rate for classifier
        if self.pretrained_config:
            # higher learning rate for classifier when using pre-trained backbone
            classifier_lr = 1e-3  # Higher than backbone LR (1e-5)
        else:
            classifier_lr = self.config.training.learning_rate
        
        optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=classifier_lr
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # early stopping variables
        best_val_auc = 0.0
        patience_counter = 0
        patience = 5  # Early stopping patience
        
        # fit energy classifier prototypes if using energy scoring
        if self.use_energy_scoring:
            self.logger.info("Fitting energy distance prototypes...")
            
            # separate normal and anomalous embeddings for prototype initialization
            train_normal_mask = self.data.y[self.train_mask] == 0
            train_anomaly_mask = self.data.y[self.train_mask] == 1
            
            if self.ensemble_energy:
                # fit prototypes for each embedding view
                normal_embedding_views = []
                anomaly_embedding_views = []
                
                for view in embedding_views:
                    view_train = view[self.train_mask.cpu()]
                    normal_embedding_views.append(view_train[train_normal_mask])
                    anomaly_embedding_views.append(view_train[train_anomaly_mask])
                
                self.energy_classifier.fit_all_prototypes(normal_embedding_views, anomaly_embedding_views)
            else:
                # fit prototypes for single energy classifier
                train_embeddings = embeddings[self.train_mask.cpu()]
                normal_embeddings = train_embeddings[train_normal_mask]
                anomaly_embeddings = train_embeddings[train_anomaly_mask]
                
                self.energy_classifier.fit_prototypes(normal_embeddings, anomaly_embeddings)
            
            self.logger.info("Energy distance prototypes fitted successfully")
        
        self.logger.info(f"Training on {len(train_features)} nodes, validating on {len(self.val_mask) if val_loader else 0} nodes")
        
        for epoch in range(1, self.config.training.epochs_finetune + 1):
            # training phase
            self.classifier.train()
            epoch_loss = 0.0
            
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                logits = self.classifier(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * x_batch.size(0)
            
            avg_loss = epoch_loss / len(train_loader.dataset)
            self.metrics_tracker.update(classification_loss=avg_loss)
            
            # validation phase
            if val_loader:
                self.classifier.eval()
                val_scores = []
                val_labels = []
                
                with torch.no_grad():
                    for x_batch, y_batch in val_loader:
                        x_batch = x_batch.to(self.device)
                        logits = self.classifier(x_batch)
                        scores = torch.sigmoid(logits).cpu().numpy()
                        val_scores.extend(scores)
                        val_labels.extend(y_batch.numpy())
                
                val_auc = roc_auc_score(val_labels, val_scores)
                self.metrics_tracker.update(val_auc=val_auc)
                
                # early stopping logic
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    # save best model
                    if self.config.training.save_checkpoints:
                        self.save_checkpoint(epoch, "best")
                else:
                    patience_counter += 1
                
                if early_stopping and patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch} (best val AUC: {best_val_auc:.4f})")
                    break
            
            self.metrics_tracker.log_epoch(epoch, "finetune")
            
            # save checkpoint
            if (self.config.training.save_checkpoints and 
                epoch % self.config.training.save_every_n_epochs == 0):
                self.save_checkpoint(epoch, "finetune")
        
        self.logger.info("Classifier fine-tuning completed")
    
    def evaluate(self, mask: torch.Tensor = None) -> Dict[str, float]:
        """Evaluate the model performance on specified mask (defaults to test_mask)"""
        if mask is None:
            if self.test_mask is None:
                raise ValueError("Test mask not loaded and no mask provided. Call load_data() first.")
            mask = self.test_mask
        
        self.logger.info(f"Starting evaluation on {len(mask)} nodes...")
        
        # generate embeddings
        if self.use_energy_scoring and self.ensemble_energy:
            embedding_views = self.extract_multiple_embeddings()
            embeddings = embedding_views[0]  # Use spatial embeddings for standard classifier
        else:
            embeddings = self.embed_full_graph()
            embedding_views = [embeddings] if self.use_energy_scoring else None
        
        # get scores based on configuration
        if self.use_energy_scoring:
            # use energy distance scoring
            self.logger.info("Computing energy distance scores...")
            
            if self.ensemble_energy:
                # use ensemble energy classifier
                test_embedding_views = [view[mask.cpu()].to(self.device) for view in embedding_views]
                with torch.no_grad():
                    energy_scores = self.energy_classifier(test_embedding_views).cpu().numpy()
            else:
                # use single energy classifier
                test_features = embeddings[mask.cpu()].to(self.device)
                with torch.no_grad():
                    energy_scores = self.energy_classifier(test_features).cpu().numpy()
            
            # use energy scores as anomaly scores
            scores = energy_scores
            
            # compute standard classifier scores for comparison
            self.classifier.eval()
            test_features = embeddings[mask.cpu()].to(self.device)
            with torch.no_grad():
                classifier_logits = self.classifier(test_features)
                classifier_scores = torch.sigmoid(classifier_logits).cpu().numpy()
            
            self.logger.info(f"Energy scores range: [{energy_scores.min():.4f}, {energy_scores.max():.4f}]")
            self.logger.info(f"Classifier scores range: [{classifier_scores.min():.4f}, {classifier_scores.max():.4f}]")
            
        else:
            # use standard classifier scores
            self.classifier.eval()
            test_features = embeddings[mask.cpu()].to(self.device)
            
            with torch.no_grad():
                logits = self.classifier(test_features)
                scores = torch.sigmoid(logits).cpu().numpy()
        
        # get true labels for test nodes
        y_true = self.data.y[mask].cpu().numpy()
        
        # compute metrics
        auc = roc_auc_score(y_true, scores)
        ap = average_precision_score(y_true, scores)
        
        results = {
            "roc_auc": auc,
            "average_precision": ap,
            "num_test_nodes": len(y_true),
            "anomaly_ratio": np.mean(y_true)
        }
        
        # log
        self.logger.info(f"Evaluation Results:")
        for metric, value in results.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return results
    
    def save_checkpoint(self, epoch: int, phase: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'model_state_dict': self.model.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'config': self.config,
            'metrics': self.metrics_tracker.metrics
        }
        
        checkpoint_path = Path(self.config.training.checkpoint_dir) / f"checkpoint_epoch_{epoch}_{phase}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
    
    def train(self):
        """Main training pipeline"""
        try:
            log_memory_usage(self.logger, self.device)
            
            # setup models
            if self.data is None:
                raise ValueError("Data not loaded. Call load_data() first.")
            
            self.setup_models()
            self.pretrain_contrastive()
            self.finetune_classifier()
            results = self.evaluate()
            
            # save
            if self.config.training.save_checkpoints:
                self.save_checkpoint(-1, "final")
            
            log_memory_usage(self.logger, self.device)
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise


def main():
    """Main training function"""
    trainer = GADTrainer()
    results = trainer.train()
    return results


if __name__ == "__main__":
    main()