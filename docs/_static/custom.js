document.addEventListener('DOMContentLoaded', function() {
    // Replace "View page source" with "Custom Name"
    document.querySelectorAll('a[href="_sources/index.rst.txt"]').forEach(function(link) {
        link.textContent = 'Edit on GitHub';
        link.href = 'https://github.com/abhi0395/qsoabsfind/blob/gh-pages/_sources/index.rst.txt';
    });

    // Add custom copyright notice
    var footer = document.querySelector('footer');
    if (footer) {
        var copyrightNotice = document.createElement('div');
        copyrightNotice.innerHTML = 'Copyright © 2021-2025 Abhijeet Anand';
        footer.appendChild(copyrightNotice);
    }
});
