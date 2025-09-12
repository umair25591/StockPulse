$(document).ready(function () {
    let uploadedFile = null;
    let stockChart = null;

    // A color palette for the clusters. You can add more colors if needed.
    const CLUSTER_COLORS = [
        '#38bdf8', // Light Blue
        '#4ade80', // Green
        '#facc15', // Yellow
        '#fb923c', // Orange
        '#f87171', // Red
        '#a78bfa', // Purple
        '#fb7185', // Pink
    ];

    // ========== Utility Functions ==========
    function enableRunButton(enabled) {
        $("#run-analysis").prop("disabled", !enabled);
    }

    function resetRunButton() {
        $("#run-analysis").prop("disabled", false).text("Run Analysis");
    }

    function setRunButtonProcessing() {
        $("#run-analysis").prop("disabled", true).text("Processing...");
    }

    function renderSummaryCards(summary) {
        const cardsHtml = `
            <div class="bg-gray-700 p-4 rounded-lg">
                <h3 class="text-sm font-medium text-gray-400">Total Rows</h3>
                <p class="mt-1 text-3xl font-semibold">${summary.rows.toLocaleString()}</p>
            </div>
            <div class="bg-gray-700 p-4 rounded-lg">
                <h3 class="text-sm font-medium text-gray-400">Clusters Found</h3>
                <p class="mt-1 text-3xl font-semibold">${summary.clusters}</p>
            </div>
            <div class="bg-gray-700 p-4 rounded-lg">
                <h3 class="text-sm font-medium text-gray-400">Anomalies Detected</h3>
                <p class="mt-1 text-3xl font-semibold">${summary.anomalies}</p>
            </div>
        `;
        $("#summary-cards").html(cardsHtml);
    }

    function renderChart(clusteredData, anomaliesData) {
        if (!clusteredData || clusteredData.length === 0) {
            alert("No data available to render the chart.");
            return;
        }

        const ctx = $("#stock-chart")[0].getContext("2d");

        // Prepare datasets for the chart
        const datasets = [];

        // 1. Base line for the closing price
        datasets.push({
            type: 'line',
            label: 'Close Price',
            data: clusteredData.map(d => ({ x: d.Date, y: d.Close })),
            borderColor: 'rgba(156, 163, 175, 0.4)', // Gray
            borderWidth: 1,
            pointRadius: 0, // No points on the base line
            tension: 0.1,
        });

        // 2. One scatter dataset for each cluster
        const uniqueClusters = [...new Set(clusteredData.map(d => d.cluster))].sort((a, b) => a - b);
        
        uniqueClusters.forEach(clusterId => {
            const clusterPoints = clusteredData
                .filter(d => d.cluster === clusterId)
                .map(d => ({ x: d.Date, y: d.Close }));

            datasets.push({
                type: 'scatter',
                label: `Cluster ${clusterId}`,
                data: clusterPoints,
                backgroundColor: CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length],
                pointRadius: 3,
            });
        });

        // 3. A separate scatter dataset for anomalies
        if (anomaliesData && anomaliesData.length > 0) {
            datasets.push({
                type: 'scatter',
                label: 'Anomaly',
                data: anomaliesData.map(d => ({ x: d.Date, y: d.Close })),
                backgroundColor: 'red',
                pointRadius: 6,
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
            });
        }
        
        // Destroy the old chart instance if it exists
        if (stockChart) {
            stockChart.destroy();
        }

        stockChart = new Chart(ctx, {
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'month',
                            tooltipFormat: 'MMM dd, yyyy',
                        },
                        title: {
                            display: true,
                            text: 'Date'
                        },
                        ticks: { color: '#9ca3af' }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Close Price'
                        },
                        ticks: { color: '#9ca3af' }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: '#e5e7eb' }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                    },
                    zoom: {
                        pan: {
                            enabled: true,
                            mode: 'x',
                        },
                        zoom: {
                            wheel: { enabled: true },
                            pinch: { enabled: true },
                            mode: 'x',
                        }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }

    // ========== Event Handlers ==========
    $("#csv-file").on("change", function (e) {
        if (e.target.files.length > 0) {
            uploadedFile = e.target.files[0];
            enableRunButton(true);
        } else {
            uploadedFile = null;
            enableRunButton(false);
        }
    });

    $("#run-analysis").on("click", function () {
        if (!uploadedFile) return alert("Please upload a CSV file first.");

        setRunButtonProcessing();

        const formData = new FormData();
        formData.append("file", uploadedFile);

        $.ajax({
            url: "/run_analysis",
            type: "POST",
            data: formData,
            processData: false,
            contentType: false,
            success: function (result) {
                console.log("Server response:", result);
                renderSummaryCards(result.summary);
                renderChart(result.results.clustered, result.results.anomalies);
            },
            error: function (xhr) {
                const errorMsg = xhr.responseJSON ? xhr.responseJSON.error : "An unknown server error occurred.";
                console.error("Server error:", errorMsg);
                alert("Server error: " + errorMsg);
            },
            complete: function () {
                resetRunButton();
            },
        });
    });
});