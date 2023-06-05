import os
import io
from create_table_fpdf2 import PDF
from PyPDF2 import PdfReader, PdfWriter


class SaveAnalyse:
    @staticmethod
    def save_analyse(column_headers, title, data, file_name="output.pdf"):
        file_path = file_name
        if os.path.isfile(file_path):
            # File exists, append new values to the existing PDF
            reader = PdfReader(file_path)
            writer = PdfWriter()

            for page in reader.pages:
                writer.add_page(page)

            pdf = PDF()
            pdf.add_page()
            pdf.set_font("Times", size=10)
            pdf.set_auto_page_break(auto=True, margin=15)
            data = list(map(str, data))
            appended_arr = [column_headers, data]
            pdf.create_table(table_data=appended_arr, title=title, cell_width="even")
            pdf.ln()

            new_page_data = pdf.output(dest="S")

            new_page_stream = io.BytesIO(new_page_data)
            new_page = PdfReader(new_page_stream).pages[0]
            writer.add_page(new_page)

            with open(file_path, "wb") as output_file:
                writer.write(output_file)
        else:
            # File doesn't exist, create a new PDF file
            pdf = PDF()
            pdf.add_page()
            pdf.set_font("Times", size=10)
            pdf.set_auto_page_break(auto=True, margin=15)
            data = list(map(str, data))
            appended_arr = [column_headers, data]
            pdf.create_table(
                table_data=appended_arr,
                title=title,
                cell_width="even",
            )
            pdf.ln()
            pdf.output(file_path)
