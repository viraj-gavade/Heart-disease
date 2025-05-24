// Form validation and UI enhancements

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips if Bootstrap is used
    if(typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    // Form validation
    const predictionForm = document.getElementById('prediction-form');
    if(predictionForm) {
        predictionForm.addEventListener('submit', function(event) {
            if (!predictionForm.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
                
                // Add was-validated class to show validation feedback
                predictionForm.classList.add('was-validated');
                
                // Scroll to the first invalid field
                const firstInvalidField = document.querySelector('.form-control:invalid');
                if(firstInvalidField) {
                    firstInvalidField.focus();
                    firstInvalidField.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }
        }, false);
    }

    // Animation for result page
    const resultSection = document.querySelector('.result-section');
    if(resultSection) {
        resultSection.classList.add('fade-in');
        
        // Animate probability bar
        const probabilityFill = document.querySelector('.probability-fill');
        if(probabilityFill) {
            const probability = parseFloat(probabilityFill.getAttribute('data-probability'));
            setTimeout(() => {
                probabilityFill.style.width = `${probability}%`;
            }, 300);
        }
    }

    // Initialize range input display
    const rangeInputs = document.querySelectorAll('input[type="range"]');
    rangeInputs.forEach(input => {
        const output = document.getElementById(input.id + '-value');
        if(output) {
            output.textContent = input.value;
            input.addEventListener('input', function() {
                output.textContent = this.value;
            });
        }
    });

    // Help popups for form fields
    const helpIcons = document.querySelectorAll('.help-icon');
    helpIcons.forEach(icon => {
        icon.addEventListener('click', function(e) {
            e.preventDefault();
            const helpText = this.getAttribute('data-help');
            alert(helpText);
        });
    });
});
