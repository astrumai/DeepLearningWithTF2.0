import os

import nbformat
from nbformat.v4.nbbase import new_markdown_cell
import re

BOOK_COMMENT = "<!--BOOK_INFORMATION-->"


BOOK_INFO = BOOK_COMMENT + """
<img align="left" style="padding-right:10px;" src="figures/Deeplearningwithtensorflow20_small_40.png">

*This notebook contains an excerpt from the [Deep Learning with Tensorflow 2.0]() by Mukesh Mithrakumar. The code is released under the [MIT license](https://opensource.org/licenses/MIT) and is available for FREE [on GitHub](https://github.com/adhiraiyan/DeepLearningWithTF2.0).*

*If you find this content useful, please consider supporting my work by [buying me a coffee](https://www.buymeacoffee.com/mmukesh)!*

"""

NOTEBOOK_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')
REG = re.compile(r'(\d\d)\.(\d\d)-(.*)\.ipynb')

def iter_notebooks():
    return sorted(nb for nb in os.listdir(NOTEBOOK_DIR) if REG.match(nb))


def add_book_info():
    for nb_name in iter_notebooks():
        nb_file = os.path.join(NOTEBOOK_DIR, nb_name)
        nb = nbformat.read(nb_file, as_version=4)

        is_comment = lambda cell: cell.source.startswith(BOOK_COMMENT)

        if is_comment(nb.cells[0]):
            print('- amending comment for {0}'.format(nb_name))
            nb.cells[0].source = BOOK_INFO
        else:
            print('- inserting comment for {0}'.format(nb_name))
            nb.cells.insert(0, new_markdown_cell(BOOK_INFO))
        nbformat.write(nb, nb_file)


if __name__ == '__main__':
    add_book_info()
