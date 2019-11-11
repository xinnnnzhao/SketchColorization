import requests
from lxml import html
from PIL import Image
from io import BytesIO

# crawl colored pic from web

base_url = "https://safebooru.org/index.php?page=post&s=list&tags=all&pid="

for i in range(0, 10000, 40):
    url = ("%s%d") % (base_url, i)
    page = requests.Session().get(base_url)
    tree = html.fromstring(page.text)
    for index, span in enumerate(tree.findall("body/div")[-1].findall("div/div")[1].findall("div")[0].findall("span")):
        pic_url = span.find("a/img").get("src")
        pic_url = pic_url.replace("thumbnails", "images").replace("thumbnail_", "")
        pic = requests.Session().get(pic_url)
        suffix = "jpg"
        if pic.status_code != 200:
            pic = requests.Session().get(pic_url.replace("jpg", "png"))
            suffix = "png"
        if pic.status_code != 200:
            continue
        image = Image.open(BytesIO(pic.content))
        image.save(('../../../resource/training_data_set/colored_data_set/%d.%s' % (i + index, suffix)))
