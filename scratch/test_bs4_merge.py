from bs4 import BeautifulSoup

html1 = "<html><body><table><thead><tr><th>H1</th></tr></thead><tbody><tr><td>Data 1</td></tr></tbody></table></body></html>"
html2 = "<html><body><table><thead><tr><th>H1</th></tr></thead><tbody><tr><td>Data 2</td></tr></tbody></table></body></html>"

soup1 = BeautifulSoup(html1, "html.parser")
soup2 = BeautifulSoup(html2, "html.parser")

tbody1 = soup1.find("tbody") or soup1.find("table")
tbody2 = soup2.find("tbody") or soup2.find("table")

rows2 = soup2.find_all("tr")

for row in rows2[1:]: # skip header
    row.extract()
    tbody1.append(row)

print(str(soup1))
