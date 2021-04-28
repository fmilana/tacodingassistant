import sys, os
import zipfile
import csv
from bs4 import BeautifulSoup

def transverse(start, end, text):
    if start == end:
        return text
    for node in start.next_siblings:
        if node == end:
            # proper end
            return text

        if node.find('w:tab'):
            text += "\t"
        node = node.find('w:t')
        if node:
            text += node.text

    # if we get here it means we did not reach the end

    # ..so go up one level
    paragraph_node = start.parent
    # TODO: check that it is a paragraph
    # print('------- paragraph_node ------------')
    # print(paragraph_node.prettify())
    # print('-----------------------------------')
    # print()

    # go to the next paragraph
    paragraph_siblings = paragraph_node.next_siblings
    next_paragraph = next(paragraph_siblings)
    while next_paragraph.name != 'w:p':
        if next_paragraph == end:
            return text
        else:
            next_paragraph = next(paragraph_siblings)
    # this is a paragraph
    next_paragraph_rows = next_paragraph.children
    first_row = next(next_paragraph_rows)
    return transverse(first_row, end, text + "\n")


def process(in_filename, out_filename):
    print('processing', in_filename)
    with zipfile.ZipFile(in_filename, 'r') as archive:
        #for name in archive.namelist():
        #    print name
        writer = csv.writer(open(out_filename, 'w'), lineterminator = '\n')
        writer.writerow(['file_name', 'comment_id', 'text', 'code'])

        doc_xml = archive.read('word/document.xml')
        #doc_soup = BeautifulSoup(doc_xml, 'xml')
        doc_soup = BeautifulSoup(doc_xml, 'lxml')
        open('tmp.xml', 'w').write(doc_soup.prettify())
        comments_xml = archive.read('word/comments.xml')
        comments_soup = BeautifulSoup(comments_xml, 'xml')

        #print 'comments:'#, comments_soup.find_all('w:comment')
        for comment in comments_soup.find_all('w:comment'):
            #print '+++'
            all_codes = comment.find_all('w:t')
            all_codes = ''.join([x.text for x in all_codes])
            all_codes = all_codes.split(';')
            all_codes = [x.strip().rstrip().lower() for x in all_codes]
            comment_id = comment['w:id']

            range_start = doc_soup.find('w:commentrangestart', attrs={"w:id": comment_id})
            range_end = doc_soup.find('w:commentrangeend', attrs={"w:id": comment_id})

            #print(comment.prettify())

            #print("+++")
            text = transverse(range_start, range_end, '')
            text = text.replace('\n', ' ')
            text = text.replace('’', "'")

            # print(in_filename)
            # print(repr(text))
            # print(all_codes)
            # print(u"\n")

            #row = [in_filename, text] + all_codes
            #writer.writerow(row)

            for code in all_codes:
                if len(code.strip()) == 0:
                    continue
                row = [in_filename, comment_id, text, code]
                writer.writerow(row)

        os.remove('tmp.xml')

src = sys.argv[1]
dst = sys.argv[2]

if os.path.isdir(src) and os.path.isdir(dst):
    # process all .docx files in the src folder
    files = [f for f in os.listdir(src) if f.endswith('.docx')]
    for f in files:
        src_file = os.path.join(src, f)
        dst_file = os.path.join(dst, f.replace('.docx', '.csv'))
        process(src_file, dst_file)
else:
    # src and dst are files
    process(src, dst)
