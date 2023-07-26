from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

"""
Overview: The purpose of this class is to prepare functions that print the
results of the program in a table format. The table is printed in a pdf file.

Maintainers: - Osman Çağlar - cglrr.osman@gmail.com
             - Abdul Hannan Ayubi - abdulhannanayubi38@gmail.com
"""


class PrintUtils:
    @staticmethod
    def print_table(column_headers, row_headers, data):
        """
        This function is used to print the results of the program in a table format.

        Args:
            column_headers (list): Column headers of the table.
            row_headers (list): Row headers of the table.
            data (list): Data of the table.
        Returns:
            None
        """
        # Calculate cell widths
        max_col_widths = [
            max(len(str(data[i][j])) for i in range(len(data)))
            for j in range(len(column_headers))
        ]

        # Set the width of the column headers
        col_header_widths = [
            max(len(header), max_col_widths[i])
            for i, header in enumerate(column_headers)
        ]

        # Print column headers (bold/bold)
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

        # Print the hyphen (bold/bold)
        print("\033[1m" + "-" * (longest_header_length) + "\033[0m")

        # Print rows and values
        for row_header, row_data in zip(row_headers, data):
            # Print row header (bold/bold)
            print(
                "\033[1m"
                + row_header.ljust(max(col_header_widths))
                + "  \033[0m"
                + "\t|\t",
                end="",
            )

            # Print values
            for i, value in enumerate(row_data):
                cell_width = col_header_widths[i]
                print(str(value).ljust(cell_width), end="\t|\t")

            print()

            # Print the hyphen (bold/bold)
            print("\033[1m" + "-" * (longest_header_length) + "\033[0m")


from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


class PDFWriter:
    def __init__(self, filename):
        self.filename = filename
        self.doc = SimpleDocTemplate(self.filename, pagesize=letter)
        self.story = []

    def _create_table(self, matrix):
        """
        This function is used to create a table to print the results of the program in a table format.

        Args:
            matrix (list): Data of the table.
        Returns:
            None
        """
        table_data = []
        style = TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )

        style.add("BACKGROUND", (0, 0), (0, -1), colors.lightgrey)
        style.add("BACKGROUND", (1, 0), (-1, 0), colors.lightgrey)

        for row in matrix:
            table_data.append(row)

        t = Table(table_data)

        t.setStyle(style)

        self.story.append(t)

    def _create_text(self, text, font_size=12, is_bold=False):
        """
        This function is used to create a text to determine the title of the table.

        Args:
            text (str): Title of the table.
            font_size (int): Font size of the title.
            is_bold (bool): Bold or not.
        Returns:
            None
        """
        styles = getSampleStyleSheet()
        style = styles["Normal"]

        if is_bold:
            style = styles["Heading1"]

        p = Paragraph(text, style)
        self.story.append(p)

    def _create_space(self, height):
        """
        This function is used to create a space between the tables.

        Args:
            height (int): Height of the space.
        Returns:
            None
        """
        spacer = Spacer(1, height)
        self.story.append(spacer)

    def add_table(self, matrix):
        self._create_table(matrix)

    def add_text(self, text, font_size=12, is_bold=False):
        self._create_text(text, font_size, is_bold)

    def save(self):
        """
        This function is used to save the pdf file.

        Args:
            None
        Returns:
            None
        """
        self.doc.build(self.story)
        print(f"PDF saved as {self.filename}")

    def add_space(self, height):
        self._create_space(height)
