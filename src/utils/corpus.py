# Taken from Denglish Corpus (https://github.com/HaifaCLG/Denglisch)

import pandas

class Corpus:
    COLUMN_NAMES = ["source", "user_name", "sen_id", "sen_num", "token", "categ"]

    def __init__(self, source):
        """Initialize corpus with data from source.

        If source is an object of type Corpus, makes a copy of its data, otherwise assumes source is the name of a CSV
        file to read data from.
        """
        if isinstance(source, type(self)):
            self._data = source._data.copy()
        else:
            self._data = pandas.read_csv(source, header=0, names=self.COLUMN_NAMES, dtype=str, na_filter=False)

    def copy(self):
        """Returns deep copy of self."""
        return Corpus(self)

    def to_csv(self, file_name, *, linesep=None):
        """Write corpus to file file_name in CSV format.

        Uses the OS default as line separator, or linesep if provided (typically "\\n" for Unix, "\\r\\n" for Windows).
        """
        if linesep != None:
            self._data.to_csv(file_name, index=False, lineterminator=linesep)
        else:
            self._data.to_csv(file_name, index=False)

    def get_tokens(self, *, index=False, sort_posts=False):
        """Get a list of all tokens and corresponding tags.

        Returns a tuple of:
          - if index is True, a list of indices (of type int) which correspond to row numbers in the CSV file
          - a list of tokens (of type str)
          - a list of tags (of type str)
        Think of it as (row numbers plus) columns "token" and "categ" from the CSV file.
        By default, the lists are sorted by index, pass sort_posts=True to sort by post id (i.e. by "sen_id") instead.
        Indices can be used to update tags using set_tag() or set_tags().
        """
        if sort_posts:
            data = self._data.sort_values(by=["sen_id", "sen_num"])
        else:
            data = self._data

        idxs = list(data.index)
        toks = list(data["token"])
        tags = list(data["categ"])

        if index:
            return idxs, toks, tags
        else:
            return toks, tags

    def get_posts(self, *, index=False, sort_posts=False):
        """For each post, get a list of its tokens and corresponding tags.

        Returns a tuple of:
          - if index is True, a list of lists of indices (of type int) which correspond to row numbers in the CSV file
          - a list of lists of tokens (of type str)
          - a list of lists of tags (of type str)
        where each sublist represents one post (i.e. its elements share the same "sen_id").
        Think of it as (row numbers plus) columns "token" and "categ" from the CSV file, grouped by "sen_id".
        By default, the lists are sorted by index, pass sort_posts=True to sort by post id (i.e. by "sen_id") instead.
        Indices can be used to update tags using set_tag() or set_tags().
        """
        if sort_posts:
            data = self._data.sort_values(by=["sen_id", "sen_num"])
        else:
            data = self._data

        idxs, toks, tags = [], [], []

        sen_ids = data["sen_id"].drop_duplicates()
        for sen_id in sen_ids:
            post = data[data["sen_id"] == sen_id]
            idxs.append(list(post.index))
            toks.append(list(post["token"]))
            tags.append(list(post["categ"]))

        if index:
            return idxs, toks, tags
        else:
            return toks, tags

    def get_sentences(self, *, index=False, sort_posts=False):
        """For each sentence, get a list of its tokens and corresponding tags.

        Returns a tuple of:
          - if index is True, a list of lists of indices (of type int) which correspond to row numbers in the CSV file
          - a list of lists of tokens (of type str)
          - a list of lists of tags (of type str)
        where each sublist represents one sentence (i.e. its elements share the same "sen_id" and "sen_num").
        Think of it as (row numbers plus) columns "token" and "categ" from the CSV file, grouped by "sen_id"/"sen_num".
        By default, the lists are sorted by index, pass sort_posts=True to sort by post id (i.e. by "sen_id") instead.
        Indices can be used to update tags using set_tag() or set_tags().
        """
        if sort_posts:
            data = self._data.sort_values(by=["sen_id", "sen_num"])
        else:
            data = self._data

        idxs, toks, tags = [], [], []

        sen_ids = data["sen_id"].drop_duplicates()
        for sen_id in sen_ids:
            post = data[data["sen_id"] == sen_id]
            sen_nums = post["sen_num"].drop_duplicates()
            for sen_num in sen_nums:
                sent = post[post["sen_num"] == sen_num]
                idxs.append(list(sent.index))
                toks.append(list(sent["token"]))
                tags.append(list(sent["categ"]))

        if index:
            return idxs, toks, tags
        else:
            return toks, tags

    def set_tag(self, idx, tag):
        """Update a single tag.

        The entry at row number idx/column "categ" is set to value tag.
        The correct row number may be obtained by passing index=True to get_tokens(), get_posts() or get_sentences().
        """
        self._data.at[idx, "categ"] = tag

    def set_tags(self, idxs, tags):
        """Update multiple tags.

        For each row number in idxs, the entry in column "categ" is set to the corresponding value in tags.
        The correct row numbers may be obtained by passing index=True to get_tokens(), get_posts() or get_sentences().
        """
        for idx, tag in zip(idxs, tags):
            self.set_tag(idx, tag)
