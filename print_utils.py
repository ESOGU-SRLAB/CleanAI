from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

class PrintUtils:
    @staticmethod
    def print_table(column_headers, row_headers, data):
        # Hücre genişliklerini hesapla
        max_col_widths = [
            max(len(str(data[i][j])) for i in range(len(data)))
            for j in range(len(column_headers))
        ]

        # Sütun başlıklarının genişliğini ayarla
        col_header_widths = [
            max(len(header), max_col_widths[i])
            for i, header in enumerate(column_headers)
        ]

        # Sütun başlıklarını yazdır (kalın/bold)
        header_row = (
            "\033[1m"
            + "\t"
            + " " * (max(col_header_widths) + 2)
            + "|\t"
            + "\t|\t".join(
                header.ljust(col_header_widths[i])
                for i, header in enumerate(column_headers)
            )
            + "\t|"
        )
        print(header_row)

        longest_header_length = len(header_row.expandtabs())

        # Kısa çizgiyi yazdır (kalın/bold)
        print("\033[1m" + "-" * (longest_header_length) + "\033[0m")

        # Satırları ve değerleri yazdır
        for row_header, row_data in zip(row_headers, data):
            # Satır başlığını yazdır (kalın/bold)
            print(
                "\033[1m"
                + row_header.ljust(max(col_header_widths))
                + "  \033[0m"
                + "\t|\t",
                end="",
            )

            # Değerleri yazdır
            for i, value in enumerate(row_data):
                cell_width = col_header_widths[i]
                print(str(value).ljust(cell_width), end="\t|\t")

            print()

            # Kısa çizgiyi yazdır (kalın/bold)
            print("\033[1m" + "-" * (longest_header_length) + "\033[0m")


from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet


class PDFTableGenerator:
    def __init__(self, pdf_title):
        self.pdf_title = pdf_title
        self.page_content = []

    def add_table(self, table_data, column_headers, row_headers, table_title, table_description):
        # Tablo boyutunu hesaplama
        num_rows = len(table_data)
        num_cols = len(table_data[0])

        # Tablo verisini düzenleme
        if len(row_headers) != num_rows:
            raise ValueError("Number of row headers does not match the number of rows in table data.")

        if len(column_headers) != num_cols:
            raise ValueError("Number of column headers does not match the number of columns in table data.")

        data = [[None] + column_headers]  # Tablo verisinin başına sütun başlıklarını ekliyoruz

        for i, row_header in enumerate(row_headers):
            if len(table_data[i]) != num_cols:
                raise ValueError("Number of values in a row does not match the number of columns in table data.")

            row = [row_header] + table_data[i]
            data.append(row)

        # Tablo oluşturma
        table = Table(data, repeatRows=1)

        # Tablo stilini belirleme
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Tüm hücrelerin dikey hizalamasını ortala
        ])

        # Sütun başlıklarını gri renkte ekleme
        table_style.add('BACKGROUND', (0, 0), (0, -1), colors.lightgrey)

        # Satır başlıklarını gri renkte ekleme
        table_style.add('BACKGROUND', (1, 0), (-1, 0), colors.lightgrey)

        table.setStyle(table_style)

        # Tablo açıklamasını oluşturma
        styles = getSampleStyleSheet()
        table_title = Paragraph(f"<font size='16'><b>{table_title}</b></font>", styles['Normal'])
        table_desc = Paragraph(f"<br/>{table_description}<br/><br/>", styles['Normal'])

        # Sayfa içeriğine tablo ve açıklamayı ekleme
        self.page_content.append(table_title)
        self.page_content.append(table_desc)
        self.page_content.append(table)

    def generate_pdf(self):
        # PDF dosyasını oluşturma
        doc = SimpleDocTemplate(self.pdf_title, pagesize=letter)

        # Sayfa içeriğini oluşturma
        content = []
        content.extend(self.page_content)

        # PDF dosyasına içeriği ekleme
        doc.build(content)

