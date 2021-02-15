import openpyxl
import unshortenit
from unshortenit import UnshortenIt
unshortener = UnshortenIt()
wb = openpyxl.load_workbook("tablerase.xlsx")
ws = wb.active
for i in range(1, 5333):
    print(i)
    try :
        link = ws["D{}".format(i)].value
        ws["D{}".format(i)] = unshortener.unshorten(link)
    except :
        ws["D{}".format(i)] = "lien non valide..."
wb.save('tablerase_2.xlsx')