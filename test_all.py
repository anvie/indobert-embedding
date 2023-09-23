import unittest
import csv
# import gpt4all_emb
import indo_bert

class TestMethods(unittest.TestCase):

    # def test_gpt4all(self):
    #     with open("./data/test.csv", "r") as f:
    #         reader = csv.reader(f)
    #         next(reader)
    #         for i, row in enumerate(reader):
    #             self._check(i, row, gpt4all_emb)

    def test_indo_bert(self):
        with open("./data/test.csv", "r") as f:
            reader = csv.reader(f, quotechar='"', delimiter=',')
            next(reader)
            for i, row in enumerate(reader):
                self._check(i, row, indo_bert)


    def _check(self, _index, row, embedding_lib):
        source = row[0]
        cmp1 = row[1]
        cmp2 = row[2]
        cmp3 = row[3]

        sim1 = embedding_lib.text_similarity(source, cmp1)
        sim2 = embedding_lib.text_similarity(source, cmp2)
        sim3 = embedding_lib.text_similarity(source, cmp3)

        # print(f"indo-bert [{_index}]: sim1 = {sim1}, sim2 = {sim2}, sim3 = {sim3}")

        sims = [(sim1, 1), (sim2, 2), (sim3, 3)]
        sims.sort(key=lambda x: x[0], reverse=True)

        print(f"[{embedding_lib.__name__}] [{_index}]: sims = {sims}")

        # ranking selalu diurutkan berdasarkan kolom 1, 2, 3
        self.assertEqual([sim[1] for sim in sims], [1,2,3])
        