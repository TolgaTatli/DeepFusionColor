# CNN ve DenseFuse Trainer classes için patch
# Bu dosyayı çalıştırarak trainer class'larını ekleyin

import os

# CNN Trainer
cnn_trainer_code = '''

class CNNFusionTrainer:
    """
    CNN Fusion Model Trainer
    """
    
    def __init__(self, num_filters=[16, 32, 64], kernel_size=3, 
                 epochs=20, batch_size=16, lr=0.001, patch_size=64,
                 pretrained_path=None):
        """
        Parametreler:
        ------------
        pretrained_path : str or None
            Pre-trained model yolu (.pth file)
        """
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patch_size = patch_size
        
        # Model oluştur
        self.model = CNNFusionNet(num_filters, kernel_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Pre-trained model yükle (varsa)
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained(pretrained_path)
            print(f"[CNN Fusion] Pre-trained model loaded from: {pretrained_path}")
        else:
            print(f"[CNN Fusion] Model created: filters={num_filters}, kernel={kernel_size}, epochs={epochs}")
    
    def train(self, train_images_ir, train_images_vis):
        """
        Modeli eğitir
        """
        print(f"[CNN Fusion] Training on {len(train_images_ir)} image pairs...")
        
        # Tüm görüntülerden patch'ler çıkar
        all_patches_ir = []
        all_patches_vis = []
        
        for ir_img, vis_img in zip(train_images_ir, train_images_vis):
            dataset = ImagePatchDataset(ir_img, vis_img, self.patch_size, self.patch_size//2)
            for i in range(len(dataset)):
                p1, p2, _ = dataset[i]
                all_patches_ir.append(p1.numpy())
                all_patches_vis.append(p2.numpy())
        
        all_patches_ir = np.array(all_patches_ir)
        all_patches_vis = np.array(all_patches_vis)
        
        # Dataset oluştur
        dataset = ImagePatchDataset(all_patches_ir.reshape(-1, self.patch_size, self.patch_size), 
                                   all_patches_vis.reshape(-1, self.patch_size, self.patch_size),
                                   self.patch_size, self.patch_size)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Loss ve optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for patch1, patch2, target in dataloader:
                patch1 = patch1.to(self.device)
                patch2 = patch2.to(self.device)
                target = target.to(self.device)
                
                fused = self.model(patch1, patch2)
                loss = criterion(fused, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")
        
        print("[CNN Fusion] Training complete!")
    
    def load_pretrained(self, model_path):
        """
        Pre-trained model yükler
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"[CNN] Loaded pre-trained model (trained on {checkpoint.get('train_samples', 'unknown')} samples)")
    
    def predict(self, img1, img2):
        """
        Inference-only
        """
        self.model.eval()
        with torch.no_grad():
            img1_tensor = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            img2_tensor = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            
            fused_tensor = self.model(img1_tensor, img2_tensor)
            fused_image = fused_tensor.squeeze().cpu().numpy()
        
        return fused_image
'''

# DenseFuse Trainer
densefuse_trainer_code = '''

class DenseFuseTrainer:
    """
    DenseFuse Model Trainer
    """
    
    def __init__(self, growth_rate=32, num_blocks=3, 
                 num_layers_per_block=4, 
                 epochs=25, batch_size=16, lr=0.0001, patch_size=64,
                 pretrained_path=None):
        """
        Parametreler:
        ------------
        pretrained_path : str or None
            Pre-trained model yolu (.pth file)
        """
        self.growth_rate = growth_rate
        self.num_blocks = num_blocks
        self.num_layers_per_block = num_layers_per_block
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patch_size = patch_size
        
        # Model oluştur
        self.model = DenseFuseNet(growth_rate, num_blocks, num_layers_per_block)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Pre-trained model yükle (varsa)
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained(pretrained_path)
            print(f"[DenseFuse] Pre-trained model loaded from: {pretrained_path}")
        else:
            print(f"[DenseFuse] Model created: growth_rate={growth_rate}, blocks={num_blocks}, layers/block={num_layers_per_block}, epochs={epochs}")
    
    def train(self, train_images_ir, train_images_vis):
        """
        Modeli eğitir
        """
        print(f"[DenseFuse] Training on {len(train_images_ir)} image pairs...")
        
        # Tüm görüntülerden patch'ler çıkar
        all_patches_ir = []
        all_patches_vis = []
        
        for ir_img, vis_img in zip(train_images_ir, train_images_vis):
            dataset = ImagePatchDataset(ir_img, vis_img, self.patch_size, self.patch_size//2)
            for i in range(len(dataset)):
                p1, p2, _ = dataset[i]
                all_patches_ir.append(p1.numpy())
                all_patches_vis.append(p2.numpy())
        
        all_patches_ir = np.array(all_patches_ir)
        all_patches_vis = np.array(all_patches_vis)
        
        # Dataset oluştur
        dataset = ImagePatchDataset(all_patches_ir.reshape(-1, self.patch_size, self.patch_size), 
                                   all_patches_vis.reshape(-1, self.patch_size, self.patch_size),
                                   self.patch_size, self.patch_size)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Loss ve optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for patch1, patch2, target in dataloader:
                patch1 = patch1.to(self.device)
                patch2 = patch2.to(self.device)
                target = target.to(self.device)
                
                fused = self.model(patch1, patch2)
                loss = criterion(fused, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")
        
        print("[DenseFuse] Training complete!")
    
    def load_pretrained(self, model_path):
        """
        Pre-trained model yükler
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"[DenseFuse] Loaded pre-trained model (trained on {checkpoint.get('train_samples', 'unknown')} samples)")
    
    def predict(self, img1, img2):
        """
        Inference-only
        """
        self.model.eval()
        with torch.no_grad():
            img1_tensor = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            img2_tensor = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            
            fused_tensor = self.model(img1_tensor, img2_tensor)
            fused_image = fused_tensor.squeeze().cpu().numpy()
        
        return fused_image
'''

print("Bu kod parçalarını ilgili dosyalara manuel olarak ekleyin:")
print("\n1. CNN Fusion Trainer -> backend/models/cnn_fusion.py")
print("\n2. DenseFuse Trainer -> backend/models/densefuse_fusion.py")
