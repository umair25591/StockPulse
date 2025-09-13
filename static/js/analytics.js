$(document).ready(function () {

    let stockChart = null;
    let analysisData = {};
    let uploadedFile = null;
    let downloadFilename = null;

    const $runAnalysisBtn = $("#run-analysis");
    const $buttonText = $("#button-text");
    const $loader = $("#loader");
    const $csvFileInput = $("#csv-file");
    const $modelSelector = $("#model-selector");
    const $thresholdSlider = $("#threshold-slider");
    const $thresholdValue = $("#threshold-value");
    const $chartTabs = $("#chart-tabs");
    const $welcomeMessage = $("#welcome-message");
    const $resultsContainer = $("#results-container");
    const $summaryCards = $("#summary-cards");
    const $stockChartCanvas = $("#stock-chart");
    const $analysisLoader = $("#analysis-loader");
    const $csvPreviewContainer = $("#csv-preview-container");
    const $csvPreviewTable = $("#csv-preview-table");
    const $downloadBtn = $("#download-csv");
    const $downloadChartBtn = $("#download-chart");


    function resetToInitialState() {
        $("#file-input").val("");
        $("#file-preview").hide();
        $("#analyze-btn").prop("disabled", true);
        $("#loader").hide();
        $("#results").hide().empty();
        $("#results-container").hide();
        $("#charts").hide().empty();
        $("#welcome-message").show();
    }


    function destroyActiveChart() {
        if (stockChart) stockChart.destroy();
    }

    function renderClusterChart() {
        destroyActiveChart();
        const { clustered, anomalies } = analysisData.results;

        stockChart = new Chart($stockChartCanvas[0].getContext("2d"), {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Stock Price (Close)',
                    data: clustered.map(d => ({ x: new Date(d.Date).valueOf(), y: d.Close })),
                    borderColor: '#22c55e',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1
                }, {
                    type: 'scatter',
                    label: 'Anomaly',
                    data: anomalies.map(d => ({ x: new Date(d.Date).valueOf(), y: d.Close })),
                    backgroundColor: 'rgba(239, 68, 68, 0.8)',
                    radius: 7,
                    borderColor: '#ffffff',
                    borderWidth: 2,
                    pointStyle: 'rectRot',
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: { unit: 'month' },
                        ticks: { color: '#9ca3af' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' }
                    },
                    y: {
                        title: { display: true, text: 'Price', color: '#9ca3af' },
                        ticks: { color: '#9ca3af' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#e5e7eb' }
                    },
                    zoom: {
                        pan: { enabled: true, mode: 'x' },
                        zoom: { wheel: { enabled: true }, mode: 'x' }
                    }
                }
            }
        });
    }

    function renderDistanceChart() {
        destroyActiveChart();
        const { clustered } = analysisData.results;
        stockChart = new Chart($stockChartCanvas[0].getContext("2d"), {
            type: 'line',
            data: {
                labels: clustered.map(d => d.Date),
                datasets: [{
                    label: 'Distance from Cluster Center',
                    data: clustered.map(d => d.distance),
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.1)',
                    fill: true,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.3
                }]
            },
            options: { responsive: true, maintainAspectRatio: false, scales: { x: { type: 'time', time: { unit: 'month' }, ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255, 255, 255, 0.05)' } }, y: { title: { display: true, text: 'Distance Metric' }, ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255, 255, 255, 0.05)' } } }, plugins: { legend: { position: 'top', labels: { color: '#e5e7eb' } }, annotation: { annotations: { thresholdLine: { type: 'line', yMin: analysisData.threshold, yMax: analysisData.threshold, borderColor: 'rgb(255, 99, 132)', borderWidth: 2, borderDash: [6, 6], label: { content: `Threshold (${analysisData.threshold})`, enabled: true, position: 'end', backgroundColor: 'rgba(255, 99, 132, 0.8)' } } } } } }
        });
    }

    function renderVolumeChart() {
        destroyActiveChart();
        const { clustered } = analysisData.results;
        stockChart = new Chart($stockChartCanvas[0].getContext("2d"), {
            type: 'bar',
            data: {
                labels: clustered.map(d => d.Date),
                datasets: [{
                    label: 'Volume',
                    data: clustered.map(d => d.Volume),
                    backgroundColor: clustered.map(d => d.Close >= d.Open ? 'rgba(16, 185, 129, 0.7)' : 'rgba(239, 68, 68, 0.7)'),
                }]
            },
            options: { responsive: true, maintainAspectRatio: false, scales: { x: { type: 'time', time: { unit: 'month' }, ticks: { color: '#9ca3af' }, grid: { display: false } }, y: { title: { display: true, text: 'Volume' }, ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255, 255, 255, 0.05)' } } }, plugins: { legend: { display: false } } }
        });
    }

    function renderDistributionChart() {
        destroyActiveChart();
        const { anomalies } = analysisData.results;

        const anomalyCounts = anomalies.reduce((acc, curr) => {
            const clusterLabel = `Cluster ${curr.cluster}`;
            acc[clusterLabel] = (acc[clusterLabel] || 0) + 1;
            return acc;
        }, {});

        const labels = Object.keys(anomalyCounts);
        const data = Object.values(anomalyCounts);

        stockChart = new Chart($stockChartCanvas[0].getContext("2d"), {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Anomalies',
                    data: data,
                    backgroundColor: [
                        'rgba(239, 68, 68, 0.8)',
                        'rgba(251, 146, 60, 0.8)',
                        'rgba(245, 158, 11, 0.8)',
                        'rgba(163, 230, 53, 0.8)',
                        'rgba(56, 189, 248, 0.8)',
                    ],
                    borderColor: '#18181b',
                    borderWidth: 4,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: { color: '#e5e7eb', font: { size: 14 } }
                    },
                    title: {
                        display: true,
                        text: 'Distribution of Anomalies by Cluster',
                        color: '#e5e7eb',
                        font: { size: 18 }
                    }
                }
            }
        });
    }

    function renderFeatureChart() {
        destroyActiveChart();
        const { clustered, anomalies } = analysisData.results;

        const featuresToCompare = ['Return', 'Volume', 'Vol_14', 'RSI_14'];

        const calculateAverages = (data, features) => {
            const sums = features.reduce((acc, f) => ({ ...acc, [f]: 0 }), {});
            data.forEach(d => {
                features.forEach(f => {
                    sums[f] += Math.abs(d[f] || 0);
                });
            });
            return features.map(f => sums[f] / data.length);
        };

        const normalAverages = calculateAverages(clustered, featuresToCompare);
        const anomalyAverages = calculateAverages(anomalies, featuresToCompare);

        const normalize = (normal, anomaly) => {
            const combined = [...normal, ...anomaly];
            const max = Math.max(...combined);
            return {
                normalNormalized: normal.map(v => v / max),
                anomalyNormalized: anomaly.map(v => v / max)
            };
        };

        const { normalNormalized, anomalyNormalized } = normalize(normalAverages, anomalyAverages);

        stockChart = new Chart($stockChartCanvas[0].getContext("2d"), {
            type: 'radar',
            data: {
                labels: featuresToCompare,
                datasets: [{
                    label: 'Normal Day (Average)',
                    data: normalNormalized,
                    backgroundColor: 'rgba(16, 185, 129, 0.2)',
                    borderColor: 'rgba(16, 185, 129, 1)',
                    borderWidth: 2,
                }, {
                    label: 'Anomalous Day (Average)',
                    data: anomalyNormalized,
                    backgroundColor: 'rgba(239, 68, 68, 0.2)',
                    borderColor: 'rgba(239, 68, 68, 1)',
                    borderWidth: 2,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        pointLabels: { color: '#e5e7eb', font: { size: 14 } },
                        ticks: {
                            backdropColor: 'transparent',
                            color: '#a1a1aa'
                        }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#e5e7eb', font: { size: 14 } } },
                    title: { display: true, text: 'Normalized Feature Comparison', color: '#e5e7eb', font: { size: 18 } }
                }
            }
        });
    }

    function renderSummaryCards(summary) {
        $summaryCards.html(`
            <div class="glass-effect p-6">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="text-sm text-zinc-400">Total Rows Processed</p>
                        <p class="text-3xl font-bold text-white mt-1">${summary.rows.toLocaleString()}</p>
                    </div>
                    <div class="p-2 bg-blue-500/20 rounded-lg">
                        <i data-lucide="database" class="text-blue-400"></i>
                    </div>
                </div>
            </div>

            <div class="glass-effect p-6">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="text-sm text-zinc-400">Clusters Found</p>
                        <p class="text-3xl font-bold text-white mt-1">${summary.clusters}</p>
                    </div>
                    <div class="p-2 bg-purple-500/20 rounded-lg">
                        <i data-lucide="layout-grid" class="text-purple-400"></i>
                    </div>
                </div>
            </div>
            
            <div class="glass-effect p-6">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="text-sm text-zinc-400">Anomalies Detected</p>
                        <p class="text-3xl font-bold text-white mt-1">${summary.anomalies}</p>
                    </div>
                    <div class="p-2 bg-red-500/20 rounded-lg">
                        <i data-lucide="siren" class="text-red-400"></i>
                    </div>
                </div>
            </div>
        `);

        if (window.lucide) {
            window.lucide.createIcons();
        }
    }

    function showError(message) {
        const errorBox = document.getElementById("error-box");
        errorBox.textContent = message;
        errorBox.style.display = "block";
    }

    function clearError() {
        const errorBox = document.getElementById("error-box");
        errorBox.textContent = "";
        errorBox.style.display = "none";
    }





    $csvFileInput.on("change", (e) => {
        uploadedFile = e.target.files.length > 0 ? e.target.files[0] : null;
        $runAnalysisBtn.prop("disabled", !uploadedFile);

        if (uploadedFile) {
            Papa.parse(uploadedFile, {
                header: false,
                preview: 51,
                complete: function (results) {
                    const data = results.data;
                    if (data.length > 0) {
                        let tableHTML = '<thead><tr>';
                        const headers = data[0];
                        headers.forEach(header => {
                            tableHTML += `<th>${header}</th>`;
                        });
                        tableHTML += '</tr></thead><tbody>';

                        for (let i = 1; i < data.length; i++) {
                            const row = data[i];
                            if (row.some(cell => cell && cell.trim() !== '')) {
                                tableHTML += '<tr>';
                                row.forEach(cell => {
                                    tableHTML += `<td>${cell}</td>`;
                                });
                                tableHTML += '</tr>';
                            }
                        }
                        tableHTML += '</tbody>';

                        $csvPreviewTable.html(tableHTML);

                        $welcomeMessage.hide();
                        $analysisLoader.addClass('hidden');
                        $resultsContainer.addClass('hidden');
                        $csvPreviewContainer.removeClass('hidden');
                    }
                }
            });
        }
    });

    $thresholdSlider.on("input", function () {
        $thresholdValue.text(parseFloat($(this).val()).toFixed(1));
    });

    $runAnalysisBtn.on("click", function () {
        if (!uploadedFile) return;
        $runAnalysisBtn.prop("disabled", true);
        $buttonText.html('Processing...');

        $resultsContainer.addClass('hidden');
        $csvPreviewContainer.addClass('hidden');

        $analysisLoader.removeClass('hidden');

        const formData = new FormData();
        formData.append("file", uploadedFile);
        formData.append("model", $modelSelector.val());
        formData.append("threshold", $thresholdSlider.val());
        formData.append("stock_symbol", $("#stock-ticker").val());

        $.ajax({
            url: "/run_analysis", type: "POST", data: formData,
            processData: false, contentType: false,
            success: (result) => {
                analysisData = result;
                downloadFilename = result.download_filename;
                $('#stock-symbol-display').text(result.stock_symbol);
                renderSummaryCards(result.summary);
                $chartTabs.find(".chart-tab").removeClass('active-tab');
                $chartTabs.find("[data-chart='cluster']").addClass('active-tab');
                renderClusterChart();
                $downloadBtn.removeClass('hidden');
                $downloadChartBtn.removeClass('hidden');
                if (window.lucide) window.lucide.createIcons();
            },
            error: (xhr) => {

                resetToInitialState();
                showError(xhr.responseJSON?.error || "An unknown server error occurred.");
            },
            complete: () => {
                if (!analysisData.error) {
                    $analysisLoader.addClass('hidden');
                    $resultsContainer.removeClass('hidden').addClass('fade-in-up');
                }
                $runAnalysisBtn.prop("disabled", false);
                $buttonText.text("Run Analysis");
            },
        });
    });

    $downloadBtn.on("click", function () {
        if (downloadFilename) {
            window.location.href = '/download_results/' + downloadFilename;
        }
    });

    $downloadChartBtn.on("click", function () {
        if (stockChart) {

            const imageUrl = stockChart.toBase64Image();

            const link = document.createElement('a');
            link.href = imageUrl;
            link.download = 'stock_analysis_chart.png';

            link.click();
        }
    });

    $chartTabs.on("click", ".chart-tab", function () {
        if (!analysisData.results) return;
        const chartType = $(this).data("chart");
        $chartTabs.find('.chart-tab').removeClass('active-tab');
        $(this).addClass('active-tab');
        if (chartType === "cluster") renderClusterChart();
        else if (chartType === "distance") renderDistanceChart();
        else if (chartType === "volume") renderVolumeChart();
        else if (chartType === "distribution") renderDistributionChart();
        else if (chartType === "features") renderFeatureChart();
    });

    $runAnalysisBtn.prop("disabled", true);
});