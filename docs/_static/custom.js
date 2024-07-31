document.addEventListener('DOMContentLoaded', function() {
    // Replace the specific HTML element in index.html
    var breadcrumbsAside = document.querySelector('li.wy-breadcrumbs-aside');
    if (breadcrumbsAside) {
        var link = breadcrumbsAside.querySelector('<a href="_sources/index.rst.txt" rel="nofollow"> View page source</a>');
        if (link) {
            link.innerHTML = "<a href='https://github.com/abhi0395/qsoabsfind/blob/main/docs/index.rst' rel="nofollow"> Edit on Github</a>";
        }
    }

    // Add custom copyright notice
    var footer = document.querySelector('footer');
    if (footer) {
        var copyrightNotice = document.createElement('div');
        copyrightNotice.innerHTML = 'Copyright © 2021-2025 Abhijeet Anand';
        footer.appendChild(copyrightNotice);
    }
});
