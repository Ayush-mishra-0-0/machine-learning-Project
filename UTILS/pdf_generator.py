import pdfkit

def generate_pdf(html_files, output_file):
    # Specify the path to wkhtmltopdf executable
    config = pdfkit.configuration(wkhtmltopdf='/usr/bin/wkhtmltopdf')  # Adjust this path based on your system
    
    # Add options to allow local file access and set page size
    options = {
        'enable-local-file-access': '',  # Allow local files like images, CSS
        'page-size': 'A4',
        'margin-top': '0.75in',
        'margin-right': '0.75in',
        'margin-bottom': '0.75in',
        'margin-left': '0.75in',
    }

    # Generate PDF from the HTML files
    pdfkit.from_file(html_files, output_file, configuration=config, options=options)

# Example usage
html_files = ['index.html']  # The HTML file(s) including image references
output_file = 'output2.pdf'   # The resulting PDF file
generate_pdf(html_files, output_file)


