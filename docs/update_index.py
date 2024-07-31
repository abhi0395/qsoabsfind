import os

def update_index_html(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Replace the "View page source" line
    new_content = content.replace(
        '<li class="wy-breadcrumbs-aside">\n'
        '            <a href="_sources/index.rst.txt" rel="nofollow"> View page source</a>\n'
        '      </li>',
        '<li class="wy-breadcrumbs-aside">\n'
        '            <a href="https://github.com/abhi0395/qsoabsfind/edit/main/docs/index.rst" rel="nofollow"> Edit on GitHub</a>\n'
        '      </li>'
    )

    # Replace the copyright line
    new_content = new_content.replace(
        'Copyright',
        'Copyright 2021-2025 Abhijeet Anand'
    )

    with open(file_path, 'w') as file:
        file.write(new_content)

if __name__ == "__main__":
    # Assuming the index.html file is in the _build/html directory
    index_html_path = os.path.join('_build', 'html', 'index.html')
    update_index_html(index_html_path)
