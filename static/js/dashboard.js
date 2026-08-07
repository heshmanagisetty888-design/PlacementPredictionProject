// JavaScript for Student Performance Dashboard

$(document).ready(function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Auto-hide alerts after 5 seconds
    setTimeout(function() {
        $('.alert').fadeOut('slow');
    }, 5000);

    // File input validation
    $('#fileInput').on('change', function() {
        var fileName = $(this).val();
        var extension = fileName.split('.').pop().toLowerCase();
        if (['csv', 'xlsx', 'xls'].indexOf(extension) === -1) {
            $(this).val('');
            alert('Please select a valid CSV or Excel file.');
        }
    });

    // AJAX request with loading spinner
    function showLoading() {
        $('#loadingSpinner').show();
    }

    function hideLoading() {
        $('#loadingSpinner').hide();
    }

    // Handle preprocessing button clicks with animation
    $('.preprocessing-btn').on('click', function() {
        $(this).addClass('active');
        setTimeout(function() {
            $('.preprocessing-btn').removeClass('active');
        }, 2000);
    });

    // Smooth scroll for navigation links
    $('a[href^="#"]').on('click', function(event) {
        var target = $(this.getAttribute('href'));
        if (target.length) {
            event.preventDefault();
            $('html, body').stop().animate({
                scrollTop: target.offset().top - 70
            }, 1000);
        }
    });

    // Handle form submissions with validation
    $('form').on('submit', function(e) {
        var form = $(this);
        if (form[0].checkValidity() === false) {
            e.preventDefault();
            e.stopPropagation();
            form.addClass('was-validated');
        }
    });

    // Update dataset info in real-time
    function updateDatasetInfo() {
        $.ajax({
            url: '/get_dataset_info',
            type: 'GET',
            success: function(response) {
                if (response.success) {
                    $('#datasetStats').html(`
                        <span class="badge bg-primary">Rows: ${response.rows}</span>
                        <span class="badge bg-success">Columns: ${response.columns}</span>
                        <span class="badge bg-info">Size: ${response.size}</span>
                    `);
                }
            }
        });
    }

    // Theme toggle (light/dark)
    $('#themeToggle').on('click', function() {
        $('body').toggleClass('dark-theme');
        var theme = $('body').hasClass('dark-theme') ? 'dark' : 'light';
        localStorage.setItem('theme', theme);
        $(this).find('i').toggleClass('bi-moon bi-sun');
    });

    // Load saved theme
    var savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        $('body').addClass('dark-theme');
        $('#themeToggle i').removeClass('bi-moon').addClass('bi-sun');
    }

    // Export functionality
    $('#exportData').on('click', function() {
        var format = $('#exportFormat').val();
        window.location.href = '/export_data?format=' + format;
    });

    // Search functionality for tables
    $('#tableSearch').on('keyup', function() {
        var value = $(this).val().toLowerCase();
        $('table tbody tr').filter(function() {
            $(this).toggle($(this).text().toLowerCase().indexOf(value) > -1);
        });
    });

    // Print functionality
    $('#printPage').on('click', function() {
        window.print();
    });

    // Keyboard shortcuts
    $(document).on('keydown', function(e) {
        // Ctrl + U = Upload
        if (e.ctrlKey && e.key === 'u') {
            e.preventDefault();
            $('#fileInput').click();
        }
        // Ctrl + P = Predict
        if (e.ctrlKey && e.key === 'p') {
            e.preventDefault();
            window.location.href = '/predict';
        }
        // Ctrl + H = Help
        if (e.ctrlKey && e.key === 'h') {
            e.preventDefault();
            window.location.href = '/help';
        }
    });

    console.log('Student Performance Dashboard loaded successfully!');
});