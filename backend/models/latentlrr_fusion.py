"""
LatLRR (Latent Low-Rank Representation) Görüntü Fusion
=======================================================
Matris ayrıştırma ve low-rank temsil kullanarak fusion yapar.
Matematiksel olarak zarif bir yöntem!

Nasıl Çalışır:
- Görüntüleri low-rank (düşük rank) ve sparse (seyrek) bileşenlere ayırır
- Low-rank: Ortak yapı/pattern
- Sparse: Benzersiz detaylar
- Bu bileşenleri akıllıca birleştirip fusion yapar
"""

import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import DictionaryLearning


class LatentLRRFusion:
    """
    Latent Low-Rank Representation tabanlı fusion
    """
    
    def __init__(self, rank_ratio=0.9, n_components=100, max_iter=30, lambda_sparse=0.1):
        """
        Parametreler:
        ------------
        rank_ratio : float (0-1 arası)
            Tutulacak singular value oranı
            Yüksek değer = daha fazla bilgi ama gürültü de artabilir
            Düşük değer = az bilgi ama temiz
            
            Etki: 0.5 = çok agresif sıkıştırma
                  0.9 = orta (önerilen)
                  0.99 = hemen hemen tüm bilgi
                  
        n_components : int
            Dictionary learning için bileşen sayısı
            Az bileşen = basit, hızlı
            Çok bileşen = karmaşık, yavaş
            
            Etki: 50 = hızlı, basit
                  100 = dengeli (önerilen)
                  200 = detaylı, yavaş
                  
        max_iter : int
            Maksimum iterasyon sayısı
            Az iterasyon = hızlı ama yetersiz öğrenme
            Çok iterasyon = yavaş ama iyi sonuç
            
        lambda_sparse : float
            Sparsity (seyreklik) parametresi
            Küçük = az sparse (daha fazla detay)
            Büyük = çok sparse (sadece önemli detaylar)
            
            Etki: 0.01 = az sparse
                  0.1 = orta (önerilen)
                  1.0 = çok sparse
        """
        self.rank_ratio = rank_ratio
        self.n_components = n_components
        self.max_iter = max_iter
        self.lambda_sparse = lambda_sparse
        
        print(f"[LatLRR Fusion] Config: rank_ratio={rank_ratio}, "
              f"n_components={n_components}, max_iter={max_iter}")
    
    def _low_rank_decomposition(self, matrix):
        """
        SVD kullanarak low-rank ayrıştırma
        
        SVD (Singular Value Decomposition):
        Matrix = U * Σ * V^T
        - U: Sol singular vektörler
        - Σ: Singular değerler (diagonal matrix)
        - V: Sağ singular vektörler
        
        Parametreler:
        ------------
        matrix : numpy.ndarray
            Ayrıştırılacak matris
            
        Returns:
        -------
        tuple : (low_rank_matrix, sparse_matrix)
        """
        # SVD uygula
        U, s, Vt = svd(matrix, full_matrices=False)
        
        # Tutulacak singular value sayısını hesapla
        # Kümülatif enerji korunumu
        energy = np.cumsum(s**2) / np.sum(s**2)
        k = np.searchsorted(energy, self.rank_ratio) + 1
        
        print(f"  SVD: Keeping {k}/{len(s)} singular values ({self.rank_ratio*100}% energy)")
        
        # Low-rank yaklaşımı oluştur
        # Sadece ilk k singular value'yu kullan
        S_k = np.zeros((U.shape[0], Vt.shape[0]))
        S_k[:k, :k] = np.diag(s[:k])
        
        low_rank = U @ S_k @ Vt
        
        # Sparse component = orijinal - low_rank
        sparse = matrix - low_rank
        
        return low_rank, sparse
    
    def _sparse_coding(self, image):
        """
        Dictionary learning ile sparse kodlama
        
        Sparse Coding: Görüntüyü az sayıda basis (dictionary) elemanının
        kombinasyonu olarak temsil eder.
        
        Parametreler:
        ------------
        image : numpy.ndarray
            Kodlanacak görüntü
            
        Returns:
        -------
        numpy.ndarray : Sparse kodlanmış görüntü
        """
        h, w = image.shape
        
        # Görüntüyü patch'lere böl (8x8)
        patch_size = 8
        patches = []
        
        for i in range(0, h - patch_size + 1, patch_size):
            for j in range(0, w - patch_size + 1, patch_size):
                patch = image[i:i+patch_size, j:j+patch_size]
                patches.append(patch.flatten())
        
        patches = np.array(patches)
        
        # Dictionary Learning uygula
        # n_components: öğrenilecek dictionary eleman sayısı
        # alpha: sparsity constraint (L1 regularization)
        dico = DictionaryLearning(
            n_components=self.n_components,
            alpha=self.lambda_sparse,
            max_iter=self.max_iter,
            fit_algorithm='cd',  # Coordinate Descent
            random_state=42
        )
        
        # Transform: sparse code'ları öğren
        sparse_code = dico.fit_transform(patches)
        
        # Reconstruct: sparse code'dan geri oluştur
        reconstructed_patches = sparse_code @ dico.components_
        
        # Patch'leri birleştir
        reconstructed = np.zeros((h, w))
        idx = 0
        for i in range(0, h - patch_size + 1, patch_size):
            for j in range(0, w - patch_size + 1, patch_size):
                reconstructed[i:i+patch_size, j:j+patch_size] = \
                    reconstructed_patches[idx].reshape(patch_size, patch_size)
                idx += 1
        
        return reconstructed[:h, :w]
    
    def fuse(self, img1, img2):
        """
        İki görüntüyü LatLRR ile birleştirir
        
        Parametreler:
        ------------
        img1, img2 : numpy.ndarray
            Birleştirilecek görüntüler
            
        Returns:
        -------
        numpy.ndarray : Füzyon edilmiş görüntü
        """
        print("[LatLRR Fusion] Starting decomposition...")
        
        # Low-rank ve sparse ayrıştırma
        lr1, sp1 = self._low_rank_decomposition(img1)
        lr2, sp2 = self._low_rank_decomposition(img2)
        
        print("[LatLRR Fusion] Fusing components...")
        
        # Low-rank bileşenleri birleştir (ortalama)
        # Low-rank: ortak yapıyı temsil eder, ortalama almak mantıklı
        fused_lr = (lr1 + lr2) / 2.0
        
        # Sparse bileşenleri birleştir (maksimum mutlak değer)
        # Sparse: benzersiz detayları temsil eder, en belirgin olanı al
        abs_sp1 = np.abs(sp1)
        abs_sp2 = np.abs(sp2)
        mask = abs_sp1 >= abs_sp2
        fused_sp = np.where(mask, sp1, sp2)
        
        # Son füzyon: low-rank + sparse
        fused_image = fused_lr + fused_sp
        
        # [0, 1] aralığına normalize et
        fused_image = np.clip(fused_image, 0, 1)
        
        return fused_image


def latentlrr_fusion(img1, img2, rank_ratio=0.9, n_components=100, 
                     max_iter=30, lambda_sparse=0.1):
    """
    Hızlı LatLRR fusion fonksiyonu
    
    Parametreler:
    ------------
    img1, img2 : numpy.ndarray
        Birleştirilecek görüntüler
    rank_ratio : float
        Singular value oranı (0-1)
    n_components : int
        Dictionary bileşen sayısı
    max_iter : int
        Maksimum iterasyon
    lambda_sparse : float
        Sparsity parametresi
        
    Returns:
    -------
    numpy.ndarray : Füzyon edilmiş görüntü
    
    Örnek Kullanım:
    --------------
    # Hızlı test
    result = latentlrr_fusion(thermal_img, visible_img, rank_ratio=0.8, 
                             n_components=50, max_iter=20)
    
    # Yüksek kalite
    result = latentlrr_fusion(thermal_img, visible_img, rank_ratio=0.95, 
                             n_components=200, max_iter=50)
    """
    fusion_model = LatentLRRFusion(rank_ratio=rank_ratio, n_components=n_components,
                                  max_iter=max_iter, lambda_sparse=lambda_sparse)
    return fusion_model.fuse(img1, img2)
