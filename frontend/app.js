/**
 * DeepFusionColor Frontend JavaScript
 * ====================================
 * Communicates with Backend API and provides UI control
 */

// API URL (backend server address)
const API_URL = 'http://localhost:5000';

// Global variables
let selectedMethod = 'wavelet';  // Default method
let image1Data = null;
let image2Data = null;
let availableMethods = [];
let metricsChart = null;

/**
 * Runs when page loads
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('DeepFusionColor Frontend started');
    
    // Add event listeners
    setupEventListeners();
    
    // Load methods
    loadMethods();
});

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Image upload
    document.getElementById('image1Input').addEventListener('change', (e) => handleImageUpload(e, 'image1'));
    document.getElementById('image2Input').addEventListener('change', (e) => handleImageUpload(e, 'image2'));
    
    // Fusion button
    document.getElementById('fusionBtn').addEventListener('click', performFusion);
}

/**
 * Load fusion methods from backend
 */
async function loadMethods() {
    try {
        const response = await fetch(`${API_URL}/methods`);
        const data = await response.json();
        availableMethods = data.methods;
        
        // Add methods to grid
        const methodGrid = document.getElementById('methodGrid');
        methodGrid.innerHTML = '';
        
        availableMethods.forEach(method => {
            const card = createMethodCard(method);
            methodGrid.appendChild(card);
        });
        
        console.log(`${availableMethods.length} methods loaded`);
    } catch (error) {
        console.error('Error loading methods:', error);
        alert('Could not connect to backend server. Please ensure the backend is running.');
    }
}

/**
 * Create a method card
 */
function createMethodCard(method) {
    const card = document.createElement('div');
    card.className = 'method-card';
    if (method.id === selectedMethod) {
        card.classList.add('selected');
    }
    
    card.innerHTML = `
        <span class="method-badge ${method.type.toLowerCase().replace(' ', '-')}">${method.type}</span>
        <h4>${method.name}</h4>
        <p>${method.description}</p>
        <div class="method-info">
            <span>⚡ ${method.speed}</span>
            <span>⭐ ${method.quality}</span>
        </div>
    `;
    
    card.dataset.methodId = method.id;
    card.addEventListener('click', () => selectMethod(method.id, card));
    
    return card;
}

/**
 * Select a method
 */
function selectMethod(methodId, clickedCard) {
    selectedMethod = methodId;
    
    // Remove selected class from all cards
    document.querySelectorAll('.method-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    // Add selected class to clicked card
    if (clickedCard) {
        clickedCard.classList.add('selected');
    } else {
        // If card not provided, find by ID
        const card = document.querySelector(`[data-method-id="${methodId}"]`);
        if (card) {
            card.classList.add('selected');
        }
    }
    
    console.log(`Method selected: ${methodId}`);
}

/**
 * Handle image upload
 */
function handleImageUpload(event, imageId) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    
    reader.onload = function(e) {
        const imgData = e.target.result;
        
        // Save to global variable
        if (imageId === 'image1') {
            image1Data = imgData;
        } else {
            image2Data = imgData;
        }
        
        // Show preview
        const previewId = imageId === 'image1' ? 'preview1' : 'preview2';
        const preview = document.getElementById(previewId);
        preview.innerHTML = `<img src="${imgData}" alt="Preview">`;
        
        console.log(`${imageId} uploaded`);
    };
    
    reader.readAsDataURL(file);
}

/**
 * Initiate fusion process
 */
async function performFusion() {
    // Validation
    if (!image1Data || !image2Data) {
        alert('Please upload both images!');
        return;
    }
    
    // Show loading
    document.getElementById('fusionBtn').disabled = true;
    document.getElementById('loadingIndicator').classList.remove('hidden');
    document.getElementById('resultsSection').classList.add('hidden');
    
    const batchMode = document.getElementById('batchModeCheckbox').checked;
    
    if (batchMode) {
        // Batch mode
        await performBatchFusion();
    } else {
        // Single fusion
        await performSingleFusion(selectedMethod);
    }
    
    // Hide loading
    document.getElementById('fusionBtn').disabled = false;
    document.getElementById('loadingIndicator').classList.add('hidden');
}

/**
 * Single fusion operation
 */
async function performSingleFusion(method) {
    try {
        console.log(`Fusion started: ${method}`);
        
        const response = await fetch(`${API_URL}/fusion`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image1: image1Data,
                image2: image2Data,
                method: method,
                params: {}  // Default parameters
            })
        });
        
        const data = await response.json();
        console.log('[DEBUG] Backend response:', data);
        console.log('[DEBUG] Response keys:', Object.keys(data));
        console.log('[DEBUG] AI Analysis field:', data.analysis);
        console.log('[DEBUG] Full response JSON:', JSON.stringify(data, null, 2));
        
        if (data.success) {
            displayResults(data);
        } else {
            alert(`Error: ${data.error}`);
        }
        
    } catch (error) {
        console.error('Fusion error:', error);
        alert('Error occurred during fusion operation!');
    }
}

/**
 * Batch mode - tests all methods
 */
async function performBatchFusion() {
    const allResults = [];
    
    for (const method of availableMethods) {
        console.log(`Batch test: ${method.name}`);
        
        try {
            const response = await fetch(`${API_URL}/fusion`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image1: image1Data,
                    image2: image2Data,
                    method: method.id,
                    params: {}
                })
            });
            
            const data = await response.json();
            console.log(`[BATCH] ${method.name} response:`, data);
            
            if (data.success) {
                console.log(`[BATCH] ${method.name} analysis:`, data.analysis);
                allResults.push({
                    method: method.name,
                    metrics: data.metrics,
                    fusedImage: data.fused_image,
                    analysis: data.analysis
                });
            }
            
        } catch (error) {
            console.error(`${method.name} error:`, error);
        }
    }
    
    console.log('[BATCH] Final allResults:', allResults);
    // Display batch results
    displayBatchResults(allResults);
}

/**
 * Display fusion results
 */
function displayResults(data) {
    console.log('[DEBUG] displayResults executed');
    // Show results section
    document.getElementById('resultsSection').classList.remove('hidden');
    
    // Show fused image
    const fusedPreview = document.getElementById('fusedImagePreview');
    fusedPreview.innerHTML = `<img src="data:image/png;base64,${data.fused_image}" alt="Fused Image">`;
    
    // Display metrics
    displayMetrics(data.metrics);
    
    // Draw chart
    drawMetricsChart(data.metrics);
    
    // Display AI Analysis
    const aiText = data.analysis || data.ai_analysis || data.aiResult || data.ai_result;
    console.log('[DEBUG] AI text value:', aiText);
    console.log('[DEBUG] AI text type:', typeof aiText);
    if (aiText) {
        console.log('[DEBUG] Calling displayAIAnalysis');
        displayAIAnalysis(aiText);
    } else {
        console.log('[DEBUG] AI analysis empty, showing fallback message');
        displayAIAnalysis('AI analysis could not be retrieved.');
    }
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

/**
 * Display AI Analysis
 */
function displayAIAnalysis(analysis) {
    console.log('[DEBUG] displayAIAnalysis started, analysis:', analysis);
    const aiAnalysisDiv = document.getElementById('aiAnalysis');
    
    if (!aiAnalysisDiv) {
        console.error('[ERROR] aiAnalysis element bulunamadı!');
        return;
    }
    console.log('[DEBUG] aiAnalysisDiv bulundu');
    
    let cardsHtml = '';
    
    if (typeof analysis === 'string') {
        console.log('[DEBUG] Analysis string tipi');
        const parsed = tryParseJson(analysis);
        if (parsed) {
            console.log('[DEBUG] JSON parse edildi');
            cardsHtml = formatAnalysisObject(parsed);
        } else {
            console.log('[DEBUG] JSON parse başarısız, gerçek string gösteriliyor');
            cardsHtml = `<div class="ai-card"><p>${escapeHtml(analysis).replace(/\n/g, '<br>')}</p></div>`;
        }
    } else if (typeof analysis === 'object') {
        console.log('[DEBUG] Analysis object tipi');
        cardsHtml = formatAnalysisObject(analysis);
    } else {
        console.log('[DEBUG] Analysis bilinmeyen tipi:', typeof analysis);
        cardsHtml = `<div class="ai-card"><p>AI analizi uygun formatta değil.</p></div>`;
    }
    
    console.log('[DEBUG] cardsHtml:', cardsHtml);
    aiAnalysisDiv.innerHTML = `
        <div class="analysis-grid">
            ${cardsHtml}
        </div>
    `;
    
    console.log('[DEBUG] AI analizi gösterildi, DOM güncellendi');
}

/**
 * Nesne tipindeki AI analiz sonuçlarını biçimlendirir
 */
function formatAnalysisObject(analysisObject) {
    if (!analysisObject || typeof analysisObject !== 'object') {
        return '<div class="ai-card"><p>AI analizi uygun formatta değil.</p></div>';
    }

    if (Array.isArray(analysisObject)) {
        return analysisObject
            .map(item => `<div class="ai-card"><p>${escapeHtml(String(item))}</p></div>`)
            .join('');
    }

    return Object.entries(analysisObject)
        .map(([key, value]) => {
            const displayValue = typeof value === 'object'
                ? escapeHtml(JSON.stringify(value, null, 2)).replace(/\n/g, '<br>')
                : escapeHtml(String(value)).replace(/\n/g, '<br>');
            return `
                <div class="ai-card">
                    <strong>${escapeHtml(capitalizeKey(key))}</strong>
                    <p>${displayValue}</p>
                </div>
            `;
        })
        .join('');
}

/**
 * AI analizi JSON stringini denemeye çalışır
 */
function tryParseJson(value) {
    try {
        return JSON.parse(value);
    } catch (err) {
        return null;
    }
}

/**
 * Metinleri güvenli hale getirir
 */
function escapeHtml(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

/**
 * JSON anahtarını başlık haline getirir
 */
function capitalizeKey(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

/**
 * Display metrics as cards
 */
function displayMetrics(metrics) {
    const metricsGrid = document.getElementById('metricsGrid');
    
    if (!metricsGrid) {
        console.error('metricsGrid element not found!');
        return;
    }
    
    metricsGrid.innerHTML = '';
    
    console.log('Displaying metrics:', metrics);
    
    // Metric definitions
    const metricDefinitions = [
        { key: 'psnr_avg', name: 'PSNR', unit: 'dB', description: 'Higher = Better' },
        { key: 'ssim_avg', name: 'SSIM', unit: '', description: 'Higher = Better' },
        { key: 'mse_avg', name: 'MSE', unit: '', description: 'Lower = Better' },
        { key: 'mi_avg', name: 'MI', unit: '', description: 'Higher = Better' },
        { key: 'entropy', name: 'Entropy', unit: 'bits', description: 'Higher = Better' },
        { key: 'sf', name: 'SF', unit: '', description: 'Higher = Better' }
    ];
    
    metricDefinitions.forEach(def => {
        const value = metrics[def.key];
        
        if (value === undefined || value === null) {
            console.warn(`Metric not found: ${def.key}`);
            return;
        }
        
        const card = document.createElement('div');
        card.className = 'metric-card';
        
        card.innerHTML = `
            <h4>${def.name}</h4>
            <div class="metric-value">${typeof value === 'number' ? value.toFixed(4) : value}</div>
            <div class="metric-description">${def.unit} ${def.description}</div>
        `;
        
        metricsGrid.appendChild(card);
    });
    
    console.log(`${metricsGrid.children.length} metric cards created`);
}

/**
 * Draw metrics comparison chart
 */
function drawMetricsChart(metrics) {
    const ctx = document.getElementById('metricsChart').getContext('2d');
    
    // Destroy previous chart if exists
    if (metricsChart) {
        metricsChart.destroy();
    }
    
    console.log('Chart çiziliyor:', metrics);
    
    // Metrikleri normalize et (0-100 arası) - TÜM METRİKLER
    const normalizedMetrics = {
        'PSNR': Math.min(100, (metrics.psnr_avg / 50) * 100),  // 50 dB = 100%
        'SSIM': metrics.ssim_avg * 100,  // Already 0-1
        'MSE': Math.max(0, Math.min(100, 100 - (metrics.mse_avg * 1000))),  // Lower is better
        'MI': Math.min(100, (metrics.mi_avg / 5) * 100),  // 5 = 100%
        'Entropy': Math.min(100, (metrics.entropy / 8) * 100),  // 8 bits = 100%
        'SF': Math.min(100, (metrics.sf / 50) * 100)  // 50 = 100%
    };
    
    // Gerçek değerleri tooltip için sakla
    const realValues = {
        'PSNR': `${metrics.psnr_avg.toFixed(2)} dB`,
        'SSIM': metrics.ssim_avg.toFixed(4),
        'MSE': metrics.mse_avg.toFixed(6),
        'MI': metrics.mi_avg.toFixed(4),
        'Entropy': `${metrics.entropy.toFixed(4)} bits`,
        'SF': metrics.sf.toFixed(4)
    };
    
    metricsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(normalizedMetrics),
            datasets: [{
                label: 'Performans Skoru (%)',
                data: Object.values(normalizedMetrics),
                backgroundColor: [
                    'rgba(102, 126, 234, 0.8)',   // PSNR - Mavi
                    'rgba(118, 75, 162, 0.8)',    // SSIM - Mor
                    'rgba(237, 100, 166, 0.8)',   // MSE - Pembe
                    'rgba(255, 154, 158, 0.8)',   // MI - Kırmızı
                    'rgba(250, 208, 196, 0.8)',   // Entropy - Turuncu
                    'rgba(154, 236, 219, 0.8)'    // SF - Turkuaz
                ],
                borderColor: [
                    'rgba(102, 126, 234, 1)',
                    'rgba(118, 75, 162, 1)',
                    'rgba(237, 100, 166, 1)',
                    'rgba(255, 154, 158, 1)',
                    'rgba(250, 208, 196, 1)',
                    'rgba(154, 236, 219, 1)'
                ],
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        },
                        font: {
                            size: 12
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 13,
                            weight: 'bold'
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Tüm Metrikler - Performans Skoru (0-100)',
                    font: {
                        size: 16,
                        weight: 'bold'
                    },
                    padding: 20
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label;
                            const score = context.parsed.y.toFixed(1);
                            const realValue = realValues[label];
                            return [
                                `Skor: ${score}%`,
                                `Değer: ${realValue}`
                            ];
                        }
                    },
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: {
                        size: 14
                    },
                    bodyFont: {
                        size: 13
                    }
                }
            }
        }
    });
    
    console.log('Chart başarıyla oluşturuldu!');
}

/**
 * Batch test sonuçlarını gösterir
 */
function displayBatchResults(results) {
    console.log('[DISPLAY_BATCH] displayBatchResults çağrıldı');
    console.log('[DISPLAY_BATCH] Results count:', results.length);
    
    // Sonuç bölümünü göster
    document.getElementById('resultsSection').classList.remove('hidden');
    
    // Tüm füzyon görüntülerini göster
    const fusedPreview = document.getElementById('fusedImagePreview');
    fusedPreview.innerHTML = '<h4>Tüm Yöntemlerle Fusion Sonuçları</h4>';
    
    results.forEach(result => {
        const container = document.createElement('div');
        container.style.display = 'inline-block';
        container.style.margin = '10px';
        container.innerHTML = `
            <p><strong>${result.method}</strong></p>
            <img src="data:image/png;base64,${result.fusedImage}" 
                 alt="${result.method}" 
                 style="max-width: 300px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        `;
        fusedPreview.appendChild(container);
    });
    
    // Display simple AI Analysis version
    console.log('[DISPLAY_BATCH] Calling displaySimpleAIAnalysis...');
    displaySimpleAIAnalysis(results);
    console.log('[DISPLAY_BATCH] displaySimpleAIAnalysis completed');
    
    // Display batch metrics table
    console.log('[DISPLAY_BATCH] Calling displayBatchMetricsTable...');
    displayBatchMetricsTable(results);
    console.log('[DISPLAY_BATCH] displayBatchMetricsTable completed');
    
    // Draw comparison chart
    drawComparisonChart(results);
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

/**
 * AI Analysis - Simple Version (Add directly to aiAnalysis div)
 */
function displaySimpleAIAnalysis(results) {
    console.log('[SIMPLE_AI] Started');
    
    const aiAnalysisDiv = document.getElementById('aiAnalysis');
    if (!aiAnalysisDiv) {
        console.error('[SIMPLE_AI] aiAnalysis div not found!');
        return;
    }
    
    console.log('[SIMPLE_AI] aiAnalysis div found');
    
    const analysisResults = results.filter(r => r.analysis && r.analysis !== null);
    console.log('[SIMPLE_AI] Methods with analysis:', analysisResults.length);
    
    if (analysisResults.length === 0) {
        console.log('[SIMPLE_AI] No analysis');
        return;
    }
    
    // Clear content
    aiAnalysisDiv.innerHTML = '';
    console.log('[SIMPLE_AI] aiAnalysis div cleared');
    
    // Create div for each analysis
    analysisResults.forEach((result, idx) => {
        console.log(`[SIMPLE_AI] Adding ${result.method} analysis (${idx + 1}/${analysisResults.length})`);
        
        const card = document.createElement('div');
        card.style.cssText = `
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        `;
        
        // Title
        const title = document.createElement('h4');
        title.textContent = `📊 ${result.method}`;
        title.style.cssText = 'margin: 0 0 15px 0; font-size: 18px; border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 10px;';
        card.appendChild(title);
        
        // Content
        const content = document.createElement('div');
        content.style.cssText = 'font-size: 14px; line-height: 1.6;';
        
        if (typeof result.analysis === 'object' && result.analysis !== null) {
            let html = '';
            Object.entries(result.analysis).forEach(([key, value]) => {
                const displayValue = typeof value === 'object'
                    ? JSON.stringify(value).substring(0, 100) + '...'
                    : String(value).substring(0, 200);
                html += `<p style="margin: 8px 0;"><strong>${capitalizeKey(key)}:</strong> ${escapeHtml(displayValue)}</p>`;
            });
            content.innerHTML = html;
        } else {
            content.textContent = String(result.analysis);
        }
        
        card.appendChild(content);
        aiAnalysisDiv.appendChild(card);
        console.log(`[SIMPLE_AI] ${result.method} added`);
    });
    
    console.log('[SIMPLE_AI] Completed');
}

/**
 * Batch test metrics comparison table
 */
function displayBatchMetricsTable(results) {
    const metricsGrid = document.getElementById('metricsGrid');
    metricsGrid.innerHTML = '';
    
    // Show separate metric cards for each method
    results.forEach((result, index) => {
        // Yöntem başlığı
        const methodTitle = document.createElement('h3');
        methodTitle.style.gridColumn = '1 / -1';
        methodTitle.style.textAlign = 'center';
        methodTitle.style.margin = '30px 0 15px 0';
        methodTitle.style.color = '#667eea';
        methodTitle.textContent = `${result.method}`;
        metricsGrid.appendChild(methodTitle);
        
        // Metrik kartlarını oluştur (tek yöntem gibi)
        const metrics = result.metrics;
        const metricCards = [
            { name: 'PSNR', value: metrics.psnr_avg.toFixed(4), unit: 'dB', desc: 'Yüksek = İyi' },
            { name: 'SSIM', value: metrics.ssim_avg.toFixed(4), unit: '', desc: 'Yüksek = İyi' },
            { name: 'MSE', value: metrics.mse_avg.toFixed(4), unit: '', desc: 'Düşük = İyi' },
            { name: 'MI', value: metrics.mi_avg.toFixed(4), unit: '', desc: 'Yüksek = İyi' },
            { name: 'Entropy', value: metrics.entropy.toFixed(4), unit: 'bits', desc: 'Yüksek = İyi' },
            { name: 'SF', value: metrics.sf.toFixed(4), unit: '', desc: 'Yüksek = İyi' }
        ];
        
        metricCards.forEach(metric => {
            const card = document.createElement('div');
            card.className = 'metric-card';
            card.innerHTML = `
                <h4>${metric.name}</h4>
                <p class="metric-value">${metric.value}</p>
                <p class="metric-unit">${metric.unit} ${metric.desc}</p>
            `;
            metricsGrid.appendChild(card);
        });
    });
    
    // Karşılaştırma tablosu başlığı
    const comparisonTitle = document.createElement('h3');
    comparisonTitle.style.gridColumn = '1 / -1';
    comparisonTitle.style.textAlign = 'center';
    comparisonTitle.style.margin = '40px 0 20px 0';
    comparisonTitle.style.color = '#667eea';
    comparisonTitle.textContent = '📊 Tüm Yöntemlerin Karşılaştırma Tablosu';
    metricsGrid.appendChild(comparisonTitle);
    
    // Tablo oluştur
    const table = document.createElement('table');
    table.style.width = '100%';
    table.style.borderCollapse = 'collapse';
    table.style.gridColumn = '1 / -1';
    table.style.background = 'white';
    table.style.borderRadius = '10px';
    table.style.overflow = 'hidden';
    table.style.boxShadow = '0 4px 6px rgba(0,0,0,0.1)';
    
    // Başlık satırı
    const thead = document.createElement('thead');
    thead.innerHTML = `
        <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
            <th style="padding: 15px; text-align: left;">Yöntem</th>
            <th style="padding: 15px; text-align: center;">PSNR (dB)</th>
            <th style="padding: 15px; text-align: center;">SSIM</th>
            <th style="padding: 15px; text-align: center;">MSE</th>
            <th style="padding: 15px; text-align: center;">MI</th>
            <th style="padding: 15px; text-align: center;">Entropy</th>
            <th style="padding: 15px; text-align: center;">SF</th>
        </tr>
    `;
    table.appendChild(thead);
    
    // Veri satırları
    const tbody = document.createElement('tbody');
    results.forEach((result, index) => {
        const m = result.metrics;
        const row = document.createElement('tr');
        row.style.background = index % 2 === 0 ? '#f8f9fa' : 'white';
        row.style.transition = 'background 0.3s';
        row.onmouseover = () => row.style.background = '#e9ecef';
        row.onmouseout = () => row.style.background = index % 2 === 0 ? '#f8f9fa' : 'white';
        
        // En iyi değerleri vurgula
        const bestPSNR = Math.max(...results.map(r => r.metrics.psnr_avg));
        const bestSSIM = Math.max(...results.map(r => r.metrics.ssim_avg));
        const bestMSE = Math.min(...results.map(r => r.metrics.mse_avg));
        const bestMI = Math.max(...results.map(r => r.metrics.mi_avg));
        const bestEntropy = Math.max(...results.map(r => r.metrics.entropy));
        const bestSF = Math.max(...results.map(r => r.metrics.sf));
        
        const highlightStyle = 'font-weight: bold; color: #667eea;';
        
        row.innerHTML = `
            <td style="padding: 12px; font-weight: 600;">${result.method}</td>
            <td style="padding: 12px; text-align: center; ${m.psnr_avg === bestPSNR ? highlightStyle : ''}">${m.psnr_avg.toFixed(2)}</td>
            <td style="padding: 12px; text-align: center; ${m.ssim_avg === bestSSIM ? highlightStyle : ''}">${m.ssim_avg.toFixed(4)}</td>
            <td style="padding: 12px; text-align: center; ${m.mse_avg === bestMSE ? highlightStyle : ''}">${m.mse_avg.toFixed(4)}</td>
            <td style="padding: 12px; text-align: center; ${m.mi_avg === bestMI ? highlightStyle : ''}">${m.mi_avg.toFixed(4)}</td>
            <td style="padding: 12px; text-align: center; ${m.entropy === bestEntropy ? highlightStyle : ''}">${m.entropy.toFixed(4)}</td>
            <td style="padding: 12px; text-align: center; ${m.sf === bestSF ? highlightStyle : ''}">${m.sf.toFixed(2)}</td>
        `;
        tbody.appendChild(row);
    });
    table.appendChild(tbody);
    
    metricsGrid.appendChild(table);
}

/**
 * Methods comparison chart
 */
function drawComparisonChart(results) {
    const ctx = document.getElementById('metricsChart').getContext('2d');
    
    if (metricsChart) {
        metricsChart.destroy();
    }
    
    const methods = results.map(r => r.method);
    const psnrData = results.map(r => r.metrics.psnr_avg);
    const ssimData = results.map(r => r.metrics.ssim_avg * 100);  // Scale to 0-100
    const miData = results.map(r => r.metrics.mi_avg * 10);  // Scale up
    
    metricsChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: methods,
            datasets: [
                {
                    label: 'PSNR',
                    data: psnrData,
                    borderColor: 'rgba(102, 126, 234, 1)',
                    backgroundColor: 'rgba(102, 126, 234, 0.2)'
                },
                {
                    label: 'SSIM (x100)',
                    data: ssimData,
                    borderColor: 'rgba(118, 75, 162, 1)',
                    backgroundColor: 'rgba(118, 75, 162, 0.2)'
                },
                {
                    label: 'MI (x10)',
                    data: miData,
                    borderColor: 'rgba(237, 100, 166, 1)',
                    backgroundColor: 'rgba(237, 100, 166, 0.2)'
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Yöntemler Arası Metrik Karşılaştırması',
                    font: { size: 16 }
                }
            }
        }
    });
}

console.log('DeepFusionColor App.js yüklendi');
