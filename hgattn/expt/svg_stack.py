import fire
from svgutils.templates import ColumnLayout 
from svgutils.transform import fromfile


def main(num_rows: int, out_file: str, *file_list: list[str]):
    layout = ColumnLayout(num_rows)
    for file in file_list:
        svg = fromfile(file)
        layout.add_figure(svg)

    layout.save(out_file)

if __name__ == "__main__":
    fire.Fire(main)
