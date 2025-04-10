let serversData = {};
let unavailableServers = [];
let selectedServers = [];
let modelsData = {};
let commonModelsData = [];
let benchmarkSteps = [];

async function loadServers() {
    const serversSelect = document.getElementById('servers');
    const loadingSpinner = document.getElementById('servers-loading');

    serversSelect.disabled = true;
    serversSelect.style.opacity = '0.5';
    loadingSpinner.style.display = 'block';

    try {
        const response = await fetch('/servers');
        if (!response.ok) throw new Error('Failed to fetch servers');
        serversData = await response.json();
        if (serversData.error) {
            serversSelect.innerHTML = '<option class="server-error">No available servers found</option>';
            unavailableServers = [];
        } else {
            unavailableServers = [];
            serversSelect.innerHTML = '';
            Object.entries(serversData).forEach(([name, config]) => {
                const option = document.createElement('option');
                option.value = name;
                option.text = `${config.label} (${config.base_url}) ${config.status === 'down' ? ' [Down]' : ''}`;
                option.className = config.status === 'down' ? 'server-down' : '';
                serversSelect.appendChild(option);
                if (config.status === 'down') unavailableServers.push(name);
                else if (selectedServers.includes(name)) {
                    option.selected = true;
                }
            });
            selectedServers = Array.from(serversSelect.selectedOptions).map(opt => opt.value);
        }
    } catch (error) {
        console.error('Error loading servers:', error);
        serversSelect.innerHTML = '<option class="server-error">Error loading servers</option>';
        unavailableServers = [];
    } finally {
        serversSelect.disabled = false;
        serversSelect.style.opacity = '1';
        loadingSpinner.style.display = 'none';
    }

    serversSelect.addEventListener('change', () => {
        selectedServers = Array.from(serversSelect.selectedOptions).map(opt => opt.value);
        updateModels();
    });
    updateModels();
    loadServerConfig();
    loadSavedConfigs();

    setInterval(checkUnavailableServers, 5000);
}

async function loadServerConfig() {
    const response = await fetch('/app_config');
    const config = await response.json();
    const serverList = document.getElementById('serverList');
    serverList.innerHTML = '';
    Object.entries(config.servers).forEach(([name, server]) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${name}</td>
            <td>${server.label}</td>
            <td>${server.base_url}</td>
            <td>${server.status === 'up' ? 'Up' : '<span class="server-down">Down</span>'}</td>
            <td><button class="btn btn-sm btn-danger" onclick="removeServer('${name}')"><i class="fas fa-trash"></i></button></td>
        `;
        serverList.appendChild(row);
    });
}

function toggleServerConfig() {
    const div = document.getElementById('serverConfigDiv');
    div.classList.toggle('hidden');
}

function addServerForm() {
    const serverList = document.getElementById('serverList');
    const row = document.createElement('tr');
    row.innerHTML = `
        <td><input type="text" class="form-control" placeholder="Name" id="newServerName"></td>
        <td><input type="text" class="form-control" placeholder="Label" id="newServerLabel"></td>
        <td><input type="text" class="form-control" placeholder="Base URL" id="newServerUrl"></td>
        <td>Pending</td>
        <td><button class="btn btn-sm btn-success" onclick="addServer(this)"><i class="fas fa-check"></i></button></td>
    `;
    serverList.appendChild(row);
}

async function addServer(button) {
    const row = button.parentElement.parentElement;
    const name = row.querySelector('#newServerName').value;
    const label = row.querySelector('#newServerLabel').value;
    const baseUrl = row.querySelector('#newServerUrl').value;
    const config = await fetch('/app_config').then(res => res.json());
    config.servers[name] = { base_url: baseUrl, label: label, status: 'down', api_call: null };
    await fetch('/app_config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
    });
    loadServerConfig();
    updateServersSelect();
}

function removeServer(name) {
    fetch('/app_config').then(res => res.json()).then(config => {
        delete config.servers[name];
        fetch('/app_config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        }).then(() => {
            loadServerConfig();
            updateServersSelect();
        });
    });
}

async function saveServerConfig() {
    toggleServerConfig();
}

async function updateServersSelect() {
    const serversSelect = document.getElementById('servers');
    const loadingSpinner = document.getElementById('servers-loading');

    serversSelect.disabled = true;
    serversSelect.style.opacity = '0.5';
    loadingSpinner.style.display = 'block';

    try {
        const response = await fetch('/servers');
        if (!response.ok) throw new Error('Failed to fetch servers');
        serversData = await response.json();
        if (serversData.error) {
            serversSelect.innerHTML = '<option class="server-error">No available servers found</option>';
            unavailableServers = [];
        } else {
            const selectedValues = Array.from(serversSelect.selectedOptions).map(opt => opt.value);
            serversSelect.innerHTML = '';
            Object.entries(serversData).forEach(([name, config]) => {
                const option = document.createElement('option');
                option.value = name;
                option.text = `${config.label} (${config.base_url}) ${config.status === 'down' ? ' [Down]' : ''}`;
                option.className = config.status === 'down' ? 'server-down' : '';
                option.selected = selectedValues.includes(name);
                serversSelect.appendChild(option);
                if (config.status === 'down') unavailableServers.push(name);
                else unavailableServers = unavailableServers.filter(s => s !== name);
            });
            selectedServers = Array.from(serversSelect.selectedOptions).map(opt => opt.value);
        }
    } catch (error) {
        console.error('Error updating servers:', error);
        serversSelect.innerHTML = '<option class="server-error">Error loading servers</option>';
        unavailableServers = [];
    } finally {
        serversSelect.disabled = false;
        serversSelect.style.opacity = '1';
        loadingSpinner.style.display = 'none';
    }

    updateModels();
}

async function checkUnavailableServers() {
    if (unavailableServers.length === 0) return;

    const serversSelect = document.getElementById('servers');
    const loadingSpinner = document.getElementById('servers-loading');

    unavailableServers.forEach(server => {
        const option = serversSelect.querySelector(`option[value="${server}"]`);
        if (option) option.disabled = true;
    });
    serversSelect.disabled = true;
    serversSelect.style.opacity = '0.5';
    loadingSpinner.style.display = 'block';

    try {
        const responses = await Promise.all(unavailableServers.map(async server => {
            const config = serversData[server];
            if (!config) return false;
            try {
                const response = await fetch(`${config.base_url}/models`, { timeout: 5000 });
                if (response.ok && (await response.json()).object === "list") {
                    return "openai";
                }
            } catch (e) {
                try {
                    const response = await fetch(`${config.base_url.replace('/v1', '')}/api/tags`, { timeout: 5000 });
                    if (response.ok && "models" in await response.json()) {
                        return "ollama";
                    }
                } catch (e) {}
            }
            return false;
        }));

        let updated = false;
        for (let i = 0; i < unavailableServers.length; i++) {
            const server = unavailableServers[i];
            const api_call = responses[i];
            if (api_call) {
                serversData[server].api_call = api_call;
                serversData[server].status = "up";
                unavailableServers.splice(i, 1);
                updated = true;
                i--;
            }
        }

        if (updated) {
            updateServersSelect();
        }
    } catch (error) {
        console.error('Error checking unavailable servers:', error);
    } finally {
        unavailableServers.forEach(server => {
            const option = serversSelect.querySelector(`option[value="${server}"]`);
            if (option) option.disabled = false;
        });
        serversSelect.disabled = false;
        serversSelect.style.opacity = '1';
        loadingSpinner.style.display = 'none';
    }
}

async function updateModels() {
    const serversSelect = document.getElementById('servers');
    const selectedServers = Array.from(serversSelect.selectedOptions).map(opt => opt.value);
    const commonModelsCheckbox = document.getElementById('commonModels');
    const modelColumns = document.getElementById('modelColumns');
    modelColumns.innerHTML = '';

    if (selectedServers.length === 0) return;

    if (commonModelsCheckbox.checked && selectedServers.length > 1) {
        const response = await fetch(`/common_models?servers=${selectedServers.join(',')}`);
        commonModelsData = await response.json();
        const div = document.createElement('div');
        div.className = 'col-md-12 mb-3';
        div.innerHTML = '<label class="form-label">Common Models</label>';
        const select = document.createElement('select');
        select.id = 'common-model-select';
        select.className = 'form-select';
        commonModelsData.forEach(model => {
            const option = document.createElement('option');
            option.value = model.internal_name;
            option.text = model.internal_name;
            select.appendChild(option);
        });
        div.appendChild(select);
        modelColumns.appendChild(div);
    } else {
        commonModelsData = [];
        selectedServers.forEach(server => {
            if (!serversData[server]) {
                console.warn(`Server ${server} not found in serversData`);
                return;
            }
            fetch(`/models?server=${server}`).then(response => {
                if (!response.ok) throw new Error(`Failed to fetch models for ${server}`);
                return response.json();
            }).then(models => {
                modelsData[server] = models;
                const div = document.createElement('div');
                div.className = 'col-md-4 mb-3';
                div.innerHTML = `<label class="form-label">${serversData[server].label}</label>`;
                const select = document.createElement('select');
                select.id = `model-${server}`;
                select.className = 'form-select';
                models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.full_name;
                    option.text = `${model.internal_name} (${model.full_name})`;
                    select.appendChild(option);
                });
                div.appendChild(select);
                modelColumns.appendChild(div);
            }).catch(error => {
                console.error(`Error fetching models for ${server}:`, error);
            });
        });
    }
}

function generateBenchmarkConfig() {
    const serversSelect = document.getElementById('servers');
    const selectedServers = Array.from(serversSelect.selectedOptions).map(opt => opt.value);
    const repeats = document.getElementById('repeats').value;
    const prompt = document.getElementById('prompt').value;
    const commonModelsCheckbox = document.getElementById('commonModels');

    if (selectedServers.length === 0) {
        alert('Please select at least one server.');
        return null;
    }

    const servers = {};
    const models = {};
    selectedServers.forEach(server => {
        if (!serversData[server]) {
            console.warn(`Server ${server} not found in serversData`);
            return;
        }
        servers[server] = { base_url: serversData[server].base_url, label: serversData[server].label };
    });

    let modelDisplay = '';
    if (commonModelsCheckbox.checked && selectedServers.length > 1 && commonModelsData.length > 0) {
        const commonModelSelect = document.getElementById('common-model-select');
        const selectedInternalName = commonModelSelect.value;
        const selectedCommonModel = commonModelsData.find(model => model.internal_name === selectedInternalName);
        if (selectedCommonModel) {
            Object.entries(selectedCommonModel.server_models).forEach(([server, modelName]) => {
                models[server] = modelName;
            });
            modelDisplay = selectedInternalName;
        }
    } else {
        let differentModels = false;
        selectedServers.forEach(server => {
            const modelSelect = document.getElementById(`model-${server}`);
            if (modelSelect) {
                models[server] = modelSelect.value;
                const internalName = modelsData[server]?.find(m => m.full_name === modelSelect.value)?.internal_name || 'Unknown';
                if (!modelDisplay) modelDisplay = internalName;
                else if (modelDisplay !== internalName) differentModels = true;
            }
        });
        if (differentModels) modelDisplay = 'Multiple Models';
    }

    return {
        servers,
        benchmarks: [{
            num_servers: selectedServers.length,
            server_combo: selectedServers.join('+'),
            models,
            custom_prompt: prompt,
            num_repeats: parseInt(repeats)
        }],
        display: {
            num_servers: selectedServers.length,
            servers: selectedServers.map(s => serversData[s]?.label || s).join(', '),
            repeats: repeats,
            model: modelDisplay,
            query: prompt
        }
    };
}

function updateJsonFromSteps() {
    const config = { servers: {}, benchmarks: [] };
    benchmarkSteps.forEach(step => {
        Object.assign(config.servers, step.servers);
        config.benchmarks.push(step.benchmarks[0]);
    });
    const configInput = document.getElementById('configInput');
    configInput.value = JSON.stringify(config, null, 2);
    updateRunJsonBtn();
}

function generateJSON() {
    const config = generateBenchmarkConfig();
    if (!config) return;
    const configInput = document.getElementById('configInput');
    configInput.value = JSON.stringify({ servers: config.servers, benchmarks: config.benchmarks }, null, 2);
    updateRunJsonBtn();
}

function addBenchmarkStep() {
    const config = generateBenchmarkConfig();
    if (!config) return;
    benchmarkSteps.push(config);
    updateBenchmarkStepsTable();
    updateJsonFromSteps();
}

function updateBenchmarkStepsTable() {
    const stepsBody = document.getElementById('benchmarkSteps');
    stepsBody.innerHTML = '';
    benchmarkSteps.forEach((step, index) => {
        const row = document.createElement('tr');
        const queryShort = step.display.query.substring(0, 50) + (step.display.query.length > 50 ? '...' : '');
        row.innerHTML = `
            <td>${step.display.num_servers}</td>
            <td>${step.display.servers}</td>
            <td>${step.display.repeats}</td>
            <td>${step.display.model}</td>
            <td>${queryShort}<span class="full-query">${step.display.query}</span></td>
            <td><button class="btn btn-sm btn-danger" onclick="removeStep(${index})"><i class="fas fa-trash"></i></button></td>
        `;
        stepsBody.appendChild(row);
    });
    document.getElementById('runMultipleBtn').disabled = benchmarkSteps.length === 0;
}

function removeStep(index) {
    benchmarkSteps.splice(index, 1);
    updateBenchmarkStepsTable();
    updateJsonFromSteps();
}

function removeAllSteps() {
    benchmarkSteps = [];
    updateBenchmarkStepsTable();
    updateJsonFromSteps();
}

async function runBenchmarkFromUI() {
    generateJSON();
    await runBenchmark();
}

async function runMultipleBenchmarks() {
    const config = { servers: {}, benchmarks: [] };
    benchmarkSteps.forEach(step => {
        Object.assign(config.servers, step.servers);
        config.benchmarks.push(step.benchmarks[0]);
    });
    const configInput = document.getElementById('configInput');
    configInput.value = JSON.stringify(config, null, 2);
    await runBenchmark();
}

async function runBenchmark() {
    const configInput = document.getElementById('configInput');
    const config = configInput.value;
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const resultsDiv = document.getElementById('results');

    try {
        const configObj = JSON.parse(config);
        progressContainer.classList.remove('hidden');
        progressBar.style.width = '0%';
        progressText.textContent = 'Starting benchmark...';
        resultsDiv.innerHTML = '';

        let pollingInterval = setInterval(async () => {
            const response = await fetch('/progress');
            const data = await response.json();
            const progress = data.total > 0 ? Math.min((data.current / data.total) * 100, 100) : 0;
            progressBar.style.width = `${progress}%`;
            progressText.textContent = `${data.message} (${Math.round(progress)}%)`;
            if (data.completed) {
                clearInterval(pollingInterval);
                if (data.results) {
                    displayResults(data.results);
                } else {
                    progressText.textContent = 'Error: ' + data.message;
                    setTimeout(() => progressContainer.classList.add('hidden'), 2000);
                }
            }
        }, 500);

        const response = await fetch('/benchmark', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ config: configObj })
        });
        if (!response.ok) throw new Error(await response.text());
    } catch (e) {
        console.error('Error:', e);
        clearInterval(pollingInterval);
        progressContainer.classList.add('hidden');
        resultsDiv.textContent = 'Error: ' + e.message;
    }
}

async function saveCurrentConfig() {
    const name = document.getElementById('configName').value;
    if (!name) {
        alert('Please enter a config name.');
        return;
    }
    const configInput = document.getElementById('configInput');
    const config = JSON.parse(configInput.value);
    await fetch('/save_config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, config })
    });
    loadSavedConfigs();
}

async function loadSavedConfigs() {
    const response = await fetch('/saved_configs');
    const data = await response.json();
    const tbody = document.getElementById('savedConfigs');
    tbody.innerHTML = '';
    data.configs.forEach(config => {
        const row = document.createElement('tr');
        const name = config.replace('.json', '');
        row.innerHTML = `
            <td>${name}</td>
            <td>
                <button class="btn btn-sm btn-primary me-1" onclick="loadConfig('${name}')"><i class="fas fa-download"></i></button>
                <button class="btn btn-sm btn-danger" onclick="deleteConfig('${name}')"><i class="fas fa-trash"></i></button>
            </td>
        `;
        tbody.appendChild(row);
    });
}

async function loadConfig(name) {
    const response = await fetch(`/load_config/${name}`);
    const config = await response.json();
    const configInput = document.getElementById('configInput');
    configInput.value = JSON.stringify(config, null, 2);
    configInput.classList.remove('hidden');
    updateRunJsonBtn();

    benchmarkSteps = [];
    if (config.benchmarks) {
        config.benchmarks.forEach(bench => {
            const servers = config.servers || {};
            const selectedServers = (bench.server_combo || '').split('+').filter(s => s);
            const models = bench.models || {};
            let modelDisplay = bench.model_name || '';
            if (!modelDisplay && Object.keys(models).length > 0) {
                const modelNames = Object.values(models);
                modelDisplay = modelNames.every(m => m === modelNames[0]) ? modelNames[0] : 'Multiple Models';
            }
            benchmarkSteps.push({
                servers,
                benchmarks: [bench],
                display: {
                    num_servers: bench.num_servers || selectedServers.length,
                    servers: selectedServers.map(s => servers[s]?.label || s).join(', '),
                    repeats: bench.num_repeats || 1,
                    model: modelDisplay,
                    query: bench.custom_prompt || 'write a 200 words story'
                }
            });
        });
        updateBenchmarkStepsTable();
    }
}

async function deleteConfig(name) {
    await fetch(`/delete_config/${name}`, { method: 'DELETE' });
    loadSavedConfigs();
}

function toggleJSON() {
    const configInput = document.getElementById('configInput');
    const runJsonBtn = document.getElementById('runJsonBtn');
    const isHidden = configInput.classList.contains('hidden');
    if (isHidden) {
        configInput.classList.remove('hidden');
        runJsonBtn.classList.remove('hidden');
    } else {
        configInput.classList.add('hidden');
        runJsonBtn.classList.add('hidden');
    }
    updateRunJsonBtn();
}

function updateRunJsonBtn() {
    const configInput = document.getElementById('configInput');
    const runJsonBtn = document.getElementById('runJsonBtn');
    runJsonBtn.disabled = !configInput.value.trim();
}

function copyToClipboard() {
    const configInput = document.getElementById('configInput');
    configInput.select();
    document.execCommand('copy');
    alert('JSON configuration copied to clipboard!');
}

function toggleContent(id) {
    const content = document.getElementById(id);
    content.classList.toggle('hidden');
}

async function displayResults(result) {
                const progressContainer = document.getElementById('progressContainer');
                const resultsDiv = document.getElementById('results');
                progressContainer.classList.add('hidden');

                let html = '<h3>Benchmark Queries Summary</h3>';
                html += '<table class="table table-striped"><thead><tr>' +
                    '<th>Benchmark</th><th>Servers</th><th>Model</th><th>Query</th><th>Repeats</th>' +
                    '</tr></thead><tbody>';
                result.summaries.forEach((summary, index) => {
                    const benchId = summary.Benchmark.match(/\d+/) ? summary.Benchmark.match(/\d+/)[0] : index + 1;
                    html += '<tr>' +
                        `<td><a href="#benchmark-${benchId}">${summary.Benchmark}</a></td>` +
                        `<td>${summary.Servers}</td>` +
                        `<td>${summary.Model}</td>` +
                        `<td>${summary.Query}</td>` +
                        `<td>${summary.Repeats}</td>` +
                        '</tr>';
                });
                html += '</tbody></table>';

                html += '<h3>Benchmark Results</h3>';
                html += `<p><a href="/export_excel/all" class="btn btn-primary">Download All Results (Excel) <small>(Unique filename generated)</small></a></p>`;
                if (result.benchmark_file) {
                    html += `<p><span class="toggle-link" onclick="toggleContent('file-content'); fetchBenchmarkFile()">+ Show Benchmark File</span></p>`;
                    html += `<div id="file-content" class="file-content hidden"><pre>Loading...</pre></div>`;
                }
                result.results.forEach(bench => {
                    const stats = bench.stats;
                    const headers = ["Benchmark", "Server", "Full Model Name", "Model Name", "Version", "Size", "Quantization",
                                    "Min Total (s)", "Max Total (s)", "Avg Total (s)", "Std Total (s)",
                                    "Min TTFT (s)", "Max TTFT (s)", "Avg TTFT (s)", "Std TTFT (s)",
                                    "Min Gen (s)", "Max Gen (s)", "Avg Gen (s)", "Std Gen (s)",
                                    "Min Tokens/s", "Max Tokens/s", "Avg Tokens/s", "Std Tokens/s",
                                    "Last Output"];

                    html += `<div class="results-table-container"><a id="benchmark-${bench.benchmark_id}"></a>`;
                    html += `<div class="accordion mb-3" id="accordion-${bench.benchmark_id}">
                                <div class="accordion-item">
                                    <h2 class="accordion-header" id="heading-${bench.benchmark_id}">
                                        <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#collapse-${bench.benchmark_id}" aria-expanded="true" aria-controls="collapse-${bench.benchmark_id}">
                                            ${bench.benchmark_id} <a href="/export_excel/${bench.benchmark_id}" class="btn btn-sm btn-primary ms-2">Download (Excel) <small>(Unique filename)</small></a>
                                        </button>
                                    </h2>
                                    <div id="collapse-${bench.benchmark_id}" class="accordion-collapse collapse show" aria-labelledby="heading-${bench.benchmark_id}" data-bs-parent="#accordion-${bench.benchmark_id}">
                                        <div class="accordion-body">`;
                    html += '<table class="table table-striped"><thead><tr>' +
                            headers.map(h => `<th>${h}</th>`).join('') +
                            '</tr></thead><tbody>';
                    bench.stats.forEach((rowTuple, idx) => {
                        const [row, lastOutput] = rowTuple;
                        const outputId = `output-${bench.benchmark_id}-${idx}`;
                        html += '<tr>' +
                            row.map(cell => `<td>${cell}</td>`).join('') +
                            `<td><span class="toggle-link" onclick="toggleContent('${outputId}')">+</span></td>` +
                            '</tr>';
                        html += `<tr id="${outputId}" class="hidden"><td colspan="${headers.length}" class="output-content"><pre>${lastOutput}</pre></td></tr>`;
                    });
                    html += '</tbody></table></div></div></div></div></div>';
                });

                resultsDiv.innerHTML = html;
            }

            async function fetchBenchmarkFile() {
                const fileContentDiv = document.getElementById('file-content');
                if (fileContentDiv.innerHTML.includes('Loading...')) {
                    const response = await fetch('/benchmark_file');
                    const data = await response.json();
                    fileContentDiv.innerHTML = `<pre>${data.content}</pre>`;
                }
            }

            document.getElementById('commonModels').addEventListener('change', updateModels);
            window.onload = loadServers;