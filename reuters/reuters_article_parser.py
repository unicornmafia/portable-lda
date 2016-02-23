import xml.etree.ElementTree


class ReutersArticle:
    def __init__(self, document_path):
        self.document_path = document_path
        self.root = xml.etree.ElementTree.parse(document_path).getroot()

    def get_all_text(self):
        text = ""
        text_roots = self.root.findall('text')
        for text_root in text_roots:
            for paragraph in text_root.findall("p"):
                text += paragraph.text + "\n"
        return text

    def get_title(self):
        title = self.root.findall('title')
        title_text = ""
        try:
            title_text = title[0].text
        except:
            print("title not found for document " + self.document_path)
        return title_text

if __name__ == '__main__':
    # test
    article = ReutersArticle("/corpora/reuters/rcv1/19970802/810919newsML.xml")
    print("TITLE:")
    print(article.get_title())
    print("TEXT:")
    print(article.get_all_text())

