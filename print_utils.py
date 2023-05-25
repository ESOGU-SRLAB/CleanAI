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


# `(input I)` is referring to the third image URI in the `uris` list, which is
# used as a test input to calculate neuron coverage and threshold coverage for
# the ResNet50 model.
