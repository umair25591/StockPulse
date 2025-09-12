$(document).ready(function () {
    let stockChart = null;
    let analysisData = {}; // Global variable to store the full API response
    let uploadedFile = null;

    // --- UI Elements ---
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

    // --- Chart Rendering Functions ---

    function destroyActiveChart() {
        if (stockChart) stockChart.destroy();
    }

    function renderClusterChart() {
        destroyActiveChart();
        const { clustered, anomalies } = analysisData.results;
        stockChart = new Chart($stockChartCanvas[0].getContext("2d"), {
            type: 'candlestick',
            data: {
                datasets: [{
                    label: 'Stock Price (OHLC)',
                    data: clustered.map(d => ({ x: new Date(d.Date).valueOf(), o: d.Open, h: d.High, l: d.Low, c: d.Close })),
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
            options: { responsive: true, maintainAspectRatio: false, scales: { x: { type: 'time', time: { unit: 'month' } }, y: { title: { display: true, text: 'Price' } } }, plugins: { legend: { labels: { color: '#e5e7eb' } }, zoom: { pan: { enabled: true, mode: 'x' }, zoom: { wheel: { enabled: true }, mode: 'x' } } } }
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

    function renderSummaryCards(summary) {
        $summaryCards.html(`
            <div class="bg-gray-800/50 p-5 rounded-lg glass-effect fade-in-up">
                <h3 class="text-sm font-medium text-gray-400">Total Rows Processed</h3>
                <p class="mt-1 text-3xl font-semibold">${summary.rows.toLocaleString()}</p>
            </div>
            <div class="bg-gray-800/50 p-5 rounded-lg glass-effect fade-in-up" style="animation-delay: 100ms;">
                <h3 class="text-sm font-medium text-gray-400">Clusters Found</h3>
                <p class="mt-1 text-3xl font-semibold">${summary.clusters}</p>
            </div>
            <div class="bg-gray-800/50 p-5 rounded-lg glass-effect fade-in-up" style="animation-delay: 200ms;">
                <h3 class="text-sm font-medium text-gray-400">Anomalies Detected</h3>
                <p class="mt-1 text-3xl font-semibold">${summary.anomalies}</p>
            </div>`);
    }

    // --- Event Handlers ---

    $csvFileInput.on("change", (e) => {
        uploadedFile = e.target.files.length > 0 ? e.target.files[0] : null;
        $runAnalysisBtn.prop("disabled", !uploadedFile);
    });

    $thresholdSlider.on("input", function() {
        $thresholdValue.text(parseFloat($(this).val()).toFixed(1));
    });

    $runAnalysisBtn.on("click", function () {
        if (!uploadedFile) return;
        $runAnalysisBtn.prop("disabled", true);
        $buttonText.text("Processing...");
        $loader.removeClass("hidden");

        const formData = new FormData();
        formData.append("file", uploadedFile);
        formData.append("model", $modelSelector.val());
        formData.append("threshold", $thresholdSlider.val());

        $.ajax({
            url: "/run_analysis", type: "POST", data: formData,
            processData: false, contentType: false,
            success: (result) => {
                analysisData = result;
                $welcomeMessage.hide();
                $resultsContainer.removeClass('hidden').addClass('fade-in-up');
                renderSummaryCards(result.summary);
                $chartTabs.find(".chart-tab").removeClass('active-tab');
                $chartTabs.find("[data-chart='cluster']").addClass('active-tab');
                renderClusterChart();
            },
            error: (xhr) => {
                alert("Error: " + (xhr.responseJSON?.error || "An unknown server error occurred."));
            },
            complete: () => {
                $runAnalysisBtn.prop("disabled", false);
                $buttonText.text("Run Analysis");
                $loader.addClass("hidden");
            },
        });
    });

    $chartTabs.on("click", ".chart-tab", function() {
        if (!analysisData.results) return;
        const chartType = $(this).data("chart");
        $chartTabs.find('.chart-tab').removeClass('active-tab');
        $(this).addClass('active-tab');
        if (chartType === "cluster") renderClusterChart();
        else if (chartType === "distance") renderDistanceChart();
        else if (chartType === "volume") renderVolumeChart();
    });

    // --- Initial State ---
    $runAnalysisBtn.prop("disabled", true);
});