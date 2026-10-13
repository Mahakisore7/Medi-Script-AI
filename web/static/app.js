document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadContent = document.querySelector('.upload-content');
    const previewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-img-btn');
    const processBtn = document.getElementById('process-btn');

    const loadingSpinner = document.getElementById('loading-spinner');
    const resultsContainer = document.getElementById('results-container');

    let selectedFile = null;

    // Drag & Drop Handlers
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadContent.style.display = 'none';
            previewContainer.style.display = 'block';
            processBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        uploadContent.style.display = 'block';
        previewContainer.style.display = 'none';
        processBtn.disabled = true;
    });

    // Process Image
    processBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI States
        processBtn.style.display = 'none';
        loadingSpinner.style.display = 'block';
        resultsContainer.style.display = 'none';
        dropZone.style.pointerEvents = 'none';

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Failed to process image');
            }

            const data = await response.json();
            renderResults(data);

            loadingSpinner.style.display = 'none';
            resultsContainer.style.display = 'block';
            processBtn.style.display = 'flex';
            processBtn.querySelector('span').innerText = 'Process Another';
            dropZone.style.pointerEvents = 'auto';

            // Scroll to results smoothly
            resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });

        } catch (error) {
            alert('Error processing prescription: ' + error.message);
            loadingSpinner.style.display = 'none';
            processBtn.style.display = 'flex';
            dropZone.style.pointerEvents = 'auto';
        }
    });

    function renderResults(data) {
        // Patient Details
        document.getElementById('res-patient-name').innerText = data.Patient?.Name || 'N/A';
        document.getElementById('res-patient-age').innerText = data.Patient?.Age || 'N/A';
        document.getElementById('res-patient-sex').innerText = data.Patient?.Sex || 'N/A';
        document.getElementById('res-date').innerText = data.Date || 'N/A';

        // Doctor & Clinic
        document.getElementById('res-doctor-name').innerText = data.DoctorName || 'N/A';
        document.getElementById('res-clinic-info').innerText = data.HospitalClinicInfo || 'N/A';

        // Medicines
        const medicinesList = document.getElementById('res-medicines-list');
        medicinesList.innerHTML = '';

        let meds = data["Medical Text"] || [];
        if (typeof meds === 'string') meds = meds.split(' | ');

        if (meds.length === 0 || (meds.length === 1 && !meds[0])) {
            medicinesList.innerHTML = '<li>No medicines detected</li>';
        } else {
            meds.forEach(med => {
                const li = document.createElement('li');
                li.innerText = med;
                medicinesList.appendChild(li);
            });
        }

        // Dosage Advice (Markdown)
        const adviceDiv = document.getElementById('res-dosage-advice');
        const rawAdvice = data["Dosage Advice"] || 'No advice generated.';
        adviceDiv.innerHTML = marked.parse(rawAdvice);

        // Other
        document.getElementById('res-other').innerText = data.Other || 'N/A';
    }
});
