# Copyright (C) 2019 Jacopo Tediosi / Peter Teichman

import collections
import itertools
import logging
import math
import operator
import os
import random
import re
import MySQLdb
from MySQLdb.constants import ER as MySQLdb_ER #https://github.com/PyMySQL/mysqlclient-python/blob/master/MySQLdb/constants/ER.py
import time
import types

from .instatrace import trace, trace_ms, trace_us
from . import scoring
from . import tokenizers

log = logging.getLogger("cobe")


class CobeError(Exception):
    pass


class Brain:
    """ The main interface for Cobe """

    # Use an empty string to denote the start/end of a chain
    END_TOKEN = ""

    # Use a magic token id for (single) whitespace, so space is never in the tokens table
    SPACE_TOKEN_ID = -1

    def __init__(self, host, port, user, password, db, **kwargs):
        """ Connect to DB and call init() method if it doesn't exist """
        db_conn = None
        try:
            db_conn = MySQLdb.connect(host=host,port=port,user=user,passwd=password,db=db,charset='utf8mb4')
        except MySQLdb._exceptions.OperationalError as e:
            if e.args[0] != MySQLdb_ER.BAD_DB_ERROR:
                raise
            log.info("Database does not exist. Creating it...")
            Brain.init(host, port, user, password, db, **kwargs)
            db_conn = MySQLdb.connect(host=host,port=port,user=user,passwd=password,db=db,charset='utf8mb4')

        with trace_us("Brain.connect_us"):
            self.graph = graph = Graph(db_conn)

        version = graph.get_info_text("version")
        if version != "2":
            raise CobeError(f"Cannot read a version {version} brain")

        self.order = int(graph.get_info_text("order"))

        self.scorer = scoring.ScorerGroup()
        self.scorer.add_scorer(1.0, scoring.CobeScorer())

        tokenizer_name = graph.get_info_text("tokenizer")
        if tokenizer_name == "MegaHAL":
            self.tokenizer = tokenizers.MegaHALTokenizer()
        else:
            self.tokenizer = tokenizers.CobeTokenizer()

        self.stemmer = None
        stemmer_name = graph.get_info_text("stemmer")

        if stemmer_name is not None:
            try:
                self.stemmer = tokenizers.CobeStemmer(stemmer_name)
                log.debug(f"Initialized a stemmer: {stemmer_name}")
            except Exception as e:
                log.error(f"Error creating stemmer: {str(e)}")

        self._end_token_id = graph.get_token_by_text(self.END_TOKEN, create=True)

        self._end_context = [self._end_token_id] * self.order
        self._end_context_id = graph.get_node_by_tokens(self._end_context)

        self._learning = False

    def start_batch_learning(self):
        """ Begin a series of batch learn operations.
        Data will not be committed to the database until stop_batch_learning is called.
        Learn text using the normal learn(text) method. """
        self._learning = True
        self.graph.drop_reply_indexes()

    def stop_batch_learning(self):
        """ Finish a series of batch learn operations """
        self._learning = False
        self.graph.commit()
        self.graph.ensure_indexes()

    def del_stemmer(self):
        self.stemmer = None

        self.graph.delete_token_stems()

        self.graph.set_info_text("stemmer", None)
        self.graph.commit()

    def set_stemmer(self, language):
        self.stemmer = tokenizers.CobeStemmer(language)

        self.graph.delete_token_stems()
        self.graph.update_token_stems(self.stemmer)

        self.graph.set_info_text("stemmer", language)
        self.graph.commit()

    def learn(self, text):
        """ Learn a string of text. If the input is not already Unicode, it will be decoded as utf-8. """
        if type(text) != str:
            # Assume that non-Unicode text is encoded as utf-8, which should be somewhat safe in the modern world
            text = text.decode("utf-8", "ignore")

        tokens = self.tokenizer.split(text)
        trace("Brain.learn_input_token_count", len(tokens))

        self._learn_tokens(tokens)

    def _to_edges(self, tokens):
        """ This is an iterator that returns the nodes of our graph:
        This is a test" -> "None This" "This is" "is a" "a test" "test None"
        Each is annotated with a boolean that tracks whether whitespace was found between the two tokens."""

        # prepend self.order Nones
        chain = self._end_context + tokens + self._end_context

        has_space = False

        context = []

        for i in range(len(chain)):
            context.append(chain[i])

            if len(context) == self.order:
                if chain[i] == self.SPACE_TOKEN_ID:
                    context.pop()
                    has_space = True
                    continue

                yield tuple(context), has_space

                context.pop(0)
                has_space = False

    def _to_graph(self, contexts):
        """ This is an iterator that returns each edge of our graph with its two nodes """
        prev = None

        for context in contexts:
            if prev is None:
                prev = context
                continue

            yield prev[0], context[1], context[0]
            prev = context

    def _learn_tokens(self, tokens):
        token_count = len([token for token in tokens if token != " "])
        if token_count < 3:
            return

        # Create each of the non-whitespace tokens
        token_ids = []
        for text in tokens:
            if text == " ":
                token_ids.append(self.SPACE_TOKEN_ID)
                continue

            token_id = self.graph.get_token_by_text(text, create=True, stemmer=self.stemmer)
            token_ids.append(token_id)

        edges = list(self._to_edges(token_ids))

        prev_id = None
        for prev, has_space, next in self._to_graph(edges):
            if prev_id is None:
                prev_id = self.graph.get_node_by_tokens(prev)
            next_id = self.graph.get_node_by_tokens(next)

            self.graph.add_edge(prev_id, next_id, has_space)
            prev_id = next_id

        if not self._learning:
            self.graph.commit()

    def reply(self, text, loop_ms=500, max_len=None):
        """ Reply to a string of text. If the input is not already Unicode, it will be decoded as utf-8 """
        if type(text) != str:
            # Assume that non-Unicode text is encoded as utf-8, which should be somewhat safe in the modern world
            text = text.decode("utf-8", "ignore")

        tokens = self.tokenizer.split(text)
        input_ids = list(map(self.graph.get_token_by_text, tokens))

        # Filter out unknown words and non-words from the potential pivots
        pivot_set = self._filter_pivots(input_ids)

        # Conflate the known ids with the stems of their words
        if self.stemmer is not None:
            self._conflate_stems(pivot_set, tokens)

        # If we didn't recognize any word tokens in the input, pick something random from the database and babble
        if len(pivot_set) == 0:
            pivot_set = self._babble()

        score_cache = {}

        best_score = -1.0
        best_reply = None

        # Loop for approximately loop_ms milliseconds.
        # This can either take more (if the first reply takes a long time to generate) or less (if the _generate_replies search ends early) time, but it should stay roughly accurate.
        start = time.time()
        end = start + loop_ms * 0.001
        count = 0

        all_replies = []


        _start = time.time()
        for edges, pivot_node in self._generate_replies(pivot_set):
            reply = Reply(self.graph, tokens, input_ids, pivot_node, edges)

            if max_len and self._too_long(max_len, reply):
                continue

            key = reply.edge_ids
            if key not in score_cache:
                with trace_us("Brain.evaluate_reply_us"):
                    score = self.scorer.score(reply)
                    score_cache[key] = score
            else:
                # Skip scoring, we've already seen this reply
                score = -1

            if score > best_score:
                best_reply = reply
                best_score = score

            # Dump all replies to the console if debugging is enabled
            if log.isEnabledFor(logging.DEBUG):
                all_replies.append((score, reply))

            count += 1
            if time.time() > end:
                break

        if best_reply is None:
            # We couldn't find any pivot words in _babble(), so we're working with an essentially empty brain.
            # Use the classic MegaHAL reply:
            return "I don't know enough to answer you yet!"

        _time = time.time() - _start

        self.scorer.end(best_reply)

        if log.isEnabledFor(logging.DEBUG):
            replies = [(score, reply.to_text()) for score, reply in all_replies]
            replies.sort()

            for score, text in replies:
                log.debug(f"{score} {text}")

        trace("Brain.reply_input_token_count", len(tokens))
        trace("Brain.known_word_token_count", len(pivot_set))

        trace("Brain.reply_us", _time)
        trace("Brain.reply_count", count, _time)
        trace("Brain.best_reply_score", int(best_score * 1000))
        trace("Brain.best_reply_length", len(best_reply.edge_ids))

        log.debug(f"Made {count} replies ({len(score_cache)} unique) in {_time} seconds")

        if len(text) > 60:
            msg = text[0:60] + "..."
        else:
            msg = text

        log.info(f"[{msg}] {count} {best_score}")

        # Look up the words for these tokens
        with trace_us("Brain.reply_words_lookup_us"):
            text = best_reply.to_text()

        return text

    def _too_long(self, max_len, reply):
        text = reply.to_text()
        if len(text) > max_len:
            log.debug(f"Over max_len [{len(text)}]: {text}")
            return True

    def _conflate_stems(self, pivot_set, tokens):
        for token in tokens:
            stem_ids = self.graph.get_token_stem_id(self.stemmer.stem(token))
            if not stem_ids:
                continue

            # Add the tuple of stems to the pivot set, and then remove the individual token_ids
            pivot_set.add(tuple(stem_ids))
            pivot_set.difference_update(stem_ids)

    def _babble(self):
        token_ids = []
        for i in range(5):
            # Generate a few random tokens that can be used as pivots
            token_id = self.graph.get_random_token()
            if token_id is not None:
                token_ids.append(token_id)
        return set(token_ids)

    def _filter_pivots(self, pivots):
        # Remove pivots that might not give good results
        tokens = set([_f for _f in pivots if _f])
        filtered = self.graph.get_word_tokens(tokens)
        if not filtered:
            filtered = self.graph.get_tokens(tokens) or []

        return set(filtered)

    def _pick_pivot(self, pivot_ids):
        pivot = random.choice(tuple(pivot_ids))

        if type(pivot) is tuple:
            # The input word was stemmed to several things
            pivot = random.choice(pivot)

        return pivot

    def _generate_replies(self, pivot_ids):
        if not pivot_ids:
            return

        end = self._end_context_id
        graph = self.graph
        search = graph.search_random_walk

        # Cache all the trailing and beginning sentences we find from each random node we search.
        # Since the node is a full n-tuple context, we can combine any pair of next_cache[node] and prev_cache[node] and get a new reply.
        next_cache = collections.defaultdict(set)
        prev_cache = collections.defaultdict(set)

        while pivot_ids:
            # Generate a reply containing one of token_ids
            pivot_id = self._pick_pivot(pivot_ids)
            node = graph.get_random_node_with_token(pivot_id)

            parts = itertools.zip_longest(search(node, end, 1), search(node, end, 0), fillvalue=None)

            for next, prev in parts:
                if next:
                    next_cache[node].add(next)
                    for p in prev_cache[node]:
                        yield p + next, node

                if prev:
                    prev = tuple(reversed(prev))
                    prev_cache[node].add(prev)
                    for n in next_cache[node]:
                        yield prev + n, node

    @staticmethod
    def init(host, port, user, password, db, order=3, tokenizer=None, **kwargs):
        """
        Initialize a brain and create the database.
        Keyword arguments:
            order -- Order of the forward/reverse Markov chains (integer)
            tokenizer -- One of Cobe, MegaHAL (default Cobe). See documentation for cobe.tokenizers for details. (string)
        """
        log.info(f"Initializing a cobe brain. Host: {host}, Port: {port}, User: {user}, Password: {password}, DB: {db}")

        if tokenizer is None:
            tokenizer = "Cobe"

        if tokenizer not in ("Cobe", "MegaHAL"):
            log.info(f"Unknown tokenizer: {tokenizer}. Using CobeTokenizer")
            tokenizer = "Cobe"
        
        # Create DB
        db_conn = MySQLdb.connect(host=host,port=port,user=user,passwd=password,charset='utf8mb4')
        db_conn.cursor().execute(f"CREATE DATABASE {db} DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_nopad_ci")
        db_conn.commit()
        db_conn.close()

        graph = Graph(MySQLdb.connect(host=host,port=port,user=user,passwd=password,db=db,charset='utf8mb4'))

        with trace_us("Brain.init_time_us"):
            graph.init(order, tokenizer)


class Reply:
    """ Provide useful support for scoring functions """
    def __init__(self, graph, tokens, token_ids, pivot_node, edge_ids):
        self.graph = graph
        self.tokens = tokens
        self.token_ids = token_ids
        self.pivot_node = pivot_node
        self.edge_ids = edge_ids
        self.text = None

    def to_text(self):
        if self.text is None:
            parts = []
            for word, has_space in map(self.graph.get_text_by_edge, self.edge_ids):
                parts.append(word)
                if has_space:
                    parts.append(" ")
            self.text = "".join(parts)
        return self.text


class Graph:
    """ A special-purpose graph class, stored in a MySQL Database """
    def __init__(self, conn, run_migrations=True):
        self._conn = conn

        if self.is_initted():
            if run_migrations:
                self._run_migrations()

            self.order = int(self.get_info_text("order"))

            self._all_tokens = ",".join([f"token{i}_id" for i in range(self.order)])
            self._all_tokens_args = " AND ".join([f"token{i}_id = %s" for i in range(self.order)])
            self._all_tokens_q = ",".join(["%s" for i in range(self.order)])
            self._last_token = f"token{self.order-1}_id"

    def cursor(self):
        return self._conn.cursor()

    def commit(self):
        with trace_us("Brain.db_commit_us"):
            self._conn.commit()

    def close(self):
        return self._conn.close()

    def is_initted(self):
        """ Return True if database is initialized (tables have already been created), False otherwise """
        try:
            self.get_info_text("order")
            return True
        except MySQLdb._exceptions.ProgrammingError as e:
            if e.args[0] != MySQLdb_ER.NO_SUCH_TABLE:
                raise
            return False

    def set_info_text(self, attribute, text):
        c = self.cursor()

        if text is None:
            c.execute("DELETE FROM info WHERE attribute = %s", (attribute,))
        else:
            rowcount = c.execute("UPDATE info SET text = %s WHERE attribute = %s", (text, attribute))
            if rowcount <= 0:
                c.execute("INSERT INTO info (attribute, text) VALUES (%s, %s)", (attribute, text))

    def get_info_text(self, attribute, default=None, text_factory=None):
        c = self.cursor()

        if text_factory is not None:
            old_text_factory = self._conn.text_factory
            self._conn.text_factory = text_factory
        
        c.execute("SELECT text FROM info WHERE attribute = %s", (attribute,))
        row = c.fetchone()

        if text_factory is not None:
            self._conn.text_factory = old_text_factory

        if row:
            return row[0]

        return default

    def get_seq_expr(self, seq):
        # Format the sequence seq as (item1, item2, item2) as appropriate for an IN () clause in SQL
        if len(seq) == 1:
            # Grab the first item from seq. Use an iterator so this works with sets as well as lists.
            return f"({next(iter(seq))})"
        r = str(tuple(seq))
        if r == "()":
            return "(-1)"
        else:
           return r

    def get_token_by_text(self, text, create=False, stemmer=None):
        c = self.cursor()
        c.execute("SELECT id FROM tokens WHERE text = %s", (text,))
        row = c.fetchone()
        if row:
            return row[0]
        elif create:
            is_word = int(bool(re.search("\w", text, re.UNICODE)))
            c.execute("INSERT INTO tokens (text, is_word) VALUES (%s, %s)", (text, is_word))

            token_id = c.lastrowid
            if stemmer is not None:
                stem = stemmer.stem(text)
                if stem is not None:
                    self.insert_stem(token_id, stem)

            return token_id

    def insert_stem(self, token_id, stem):
        self.cursor().execute("INSERT INTO token_stems (token_id, stem) VALUES (%s, %s)", (token_id, stem))

    def get_token_stem_id(self, stem):
        c = self.cursor()
        c.execute("SELECT token_id FROM token_stems WHERE token_stems.stem = %s", (stem,))
        rows = c.fetchall()
        if rows:
            return list(map(operator.itemgetter(0), rows))

    def get_word_tokens(self, token_ids):
        c = self.cursor()
        c.execute(f"SELECT id FROM tokens WHERE id IN {self.get_seq_expr(token_ids)} AND is_word = 1")
        rows = c.fetchall()
        if rows:
            return list(map(operator.itemgetter(0), rows))

    def get_tokens(self, token_ids):
        c = self.cursor()
        c.execute(f"SELECT id FROM tokens WHERE id IN {self.get_seq_expr(token_ids)}")
        rows = c.fetchall()
        if rows:
            return list(map(operator.itemgetter(0), rows))

    def get_node_by_tokens(self, tokens):
        c = self.cursor()

        c.execute(f"SELECT id FROM nodes WHERE {self._all_tokens_args}", tokens)
        row = c.fetchone()
        if row:
            return int(row[0])

        # If not found, create the node
        c.execute(f"INSERT INTO nodes (count, {self._all_tokens})  VALUES (0, {self._all_tokens_q})", tokens)
        return c.lastrowid

    def get_text_by_edge(self, edge_id):
        c = self.cursor()
        c.execute(f"SELECT tokens.text, edges.has_space FROM nodes, edges, tokens WHERE edges.id = %s AND edges.prev_node = nodes.id AND nodes.{self._last_token} = tokens.id", (edge_id,))
        return c.fetchone()

    def get_random_token(self):
        c = self.cursor()
        # Token 1 is the end_token_id, so we want to generate a random token id from 2..max(id) inclusive.
        c.execute("SELECT id FROM tokens WHERE id>1 ORDER BY RAND() LIMIT 1")
        row = c.fetchone()
        if row:
            return row[0]

    def get_random_node_with_token(self, token_id):
        c = self.cursor()
        c.execute("SELECT id FROM nodes WHERE token0_id = %s ORDER BY RAND() LIMIT 1", (token_id,))
        row = c.fetchone()
        if row:
            return int(row[0])

    def get_edge_logprob(self, edge_id):
        # Each edge goes from an n-gram node (word1, word2, word3) to another (word2, word3, word4).
        # Calculate the probability: P(word4|word1,word2,word3) = count(edge_id) / count(prev_node_id)

        c = self.cursor()

        c.execute("SELECT edges.count, nodes.count FROM edges, nodes WHERE edges.id = %s AND edges.prev_node = nodes.id", (edge_id,))

        edge_count, node_count = c.fetchone()
        return math.log(edge_count, 2) - math.log(node_count, 2)

    def has_space(self, edge_id):
        c = self.cursor()
        c.execute("SELECT has_space FROM edges WHERE id = %s", (edge_id,))
        row = c.fetchone()
        if row:
            return bool(row[0])

    def add_edge(self, prev_node, next_node, has_space):
        # The count on the next_node in the nodes table must be incremented here, to register that the node has been seen an additional time.
        # This is now handled by database triggers.
        c = self.cursor()

        assert type(has_space) == bool

        args = (prev_node, next_node, int(has_space))

        c.execute("UPDATE edges SET count = count + 1 WHERE prev_node = %s AND next_node = %s AND has_space = %s", args)
        if c.rowcount == 0:
            c.execute("INSERT INTO edges (prev_node, next_node, has_space, count) VALUES (%s, %s, %s, 1)", args)

    def search_bfs(self, start_id, end_id, direction):
        if direction:
            q = "SELECT id, next_node FROM edges WHERE prev_node = %s"
        else:
            q = "SELECT id, prev_node FROM edges WHERE next_node = %s"

        c = self.cursor()

        left = collections.deque([(start_id, tuple())])
        while left:
            cur, path = left.popleft()
            c.execute(q, (cur,))
            rows = c.fetchall()

            for rowid, next in rows:
                newpath = path + (rowid,)

                if next == end_id:
                    yield newpath
                else:
                    left.append((next, newpath))

    def search_random_walk(self, start_id, end_id, direction):
        """ Walk once randomly from start_id to end_id """
        if direction:
            q = "SELECT id, next_node FROM edges WHERE prev_node = %s ORDER BY RAND() LIMIT 1"
        else:
            q = "SELECT id, prev_node FROM edges WHERE next_node = %s ORDER BY RAND() LIMIT 1"

        c = self.cursor()

        left = collections.deque([(start_id, tuple())])
        while left:
            cur, path = left.popleft()
            c.execute(q, (cur,))
            rows = c.fetchall()

            # Note: the LIMIT 1 above means this list only contains one row.
            # Using a list here so this matches the bfs() code, so the two functions can be more easily combined later.
            for rowid, next in rows:
                newpath = path + (rowid,)
                if next == end_id:
                    yield newpath
                else:
                    left.append((next, newpath))

    def init(self, order, tokenizer, run_migrations=True):
        """ Create tables """
        c = self.cursor()

        log.debug("Creating table: info")
        c.execute("CREATE TABLE IF NOT EXISTS info (attribute VARCHAR(30) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_nopad_ci NOT NULL, text VARCHAR(30) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_nopad_ci NOT NULL, PRIMARY KEY(attribute)) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_nopad_ci")

        log.debug("Creating table: tokens")
        c.execute("CREATE TABLE IF NOT EXISTS tokens (id BIGINT PRIMARY KEY AUTO_INCREMENT, text TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_nopad_ci NOT NULL, INDEX(text(6)), is_word TINYINT(1) NOT NULL) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_nopad_ci")

        tokens = []
        for i in range(order):
            tokens.append(f"token{i}_id BIGINT REFERENCES token(id)")

        log.debug("Creating table: token_stems")
        c.execute("CREATE TABLE token_stems (token_id INT, stem TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_nopad_ci NOT NULL) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_nopad_ci")

        log.debug("Creating table: nodes")
        c.execute(f"CREATE TABLE nodes (id BIGINT PRIMARY KEY AUTO_INCREMENT, count INTEGER NOT NULL, {', '.join(tokens)}) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_nopad_ci")

        log.debug("Creating table: edges")
        c.execute("CREATE TABLE edges (id BIGINT PRIMARY KEY AUTO_INCREMENT, prev_node BIGINT NOT NULL REFERENCES nodes(id), next_node BIGINT NOT NULL REFERENCES nodes(id), count INTEGER NOT NULL, has_space TINYINT(1) NOT NULL) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_nopad_ci")

        if run_migrations:
            self._run_migrations()

        # Save the order of this brain
        self.set_info_text("order", str(order))
        self.order = order

        # Save the tokenizer
        self.set_info_text("tokenizer", tokenizer)

        # Save the brain/schema version
        self.set_info_text("version", "2")

        self.commit()
        self.ensure_indexes()

        self.close()

    def drop_reply_indexes(self):
        c = self.cursor()

        try:
            c.execute("DROP INDEX edges_all_next ON edges")
        except MySQLdb._exceptions.OperationalError as e:
            if e.args[0] != MySQLdb_ER.CANT_DROP_FIELD_OR_KEY:
                raise

        try:
            c.execute("DROP INDEX edges_all_prev ON edges")
        except MySQLdb._exceptions.OperationalError as e:
            if e.args[0] != MySQLdb_ER.CANT_DROP_FIELD_OR_KEY:
                raise

        try:
            c.execute("CREATE INDEX learn_index ON edges (prev_node, next_node)")
        except MySQLdb._exceptions.OperationalError as e:
            if e.args[0] != MySQLdb_ER.DUP_KEYNAME:
                raise

    def ensure_indexes(self):
        c = self.cursor()
        
        # Remove the temporary learning index if it exists
        try:
            c.execute("DROP INDEX learn_index ON edges")
        except MySQLdb._exceptions.OperationalError as e:
            if e.args[0] != MySQLdb_ER.CANT_DROP_FIELD_OR_KEY:
                raise

        token_ids = ", ".join([f"token{i}_id" for i in range(self.order)])
        try:
            c.execute(f"CREATE UNIQUE INDEX nodes_token_ids on nodes ({token_ids})")
        except MySQLdb._exceptions.OperationalError as e:
            if e.args[0] != MySQLdb_ER.DUP_KEYNAME:
                raise

        try:
            c.execute("CREATE UNIQUE INDEX edges_all_next ON edges (next_node, prev_node, has_space, count)")
        except MySQLdb._exceptions.OperationalError as e:
            if e.args[0] != MySQLdb_ER.DUP_KEYNAME:
                raise

        try:
            c.execute("CREATE UNIQUE INDEX edges_all_prev ON edges (prev_node, next_node, has_space, count)")
        except MySQLdb._exceptions.OperationalError as e:
            if e.args[0] != MySQLdb_ER.DUP_KEYNAME:
                raise

    def delete_token_stems(self):
        c = self.cursor()

        # Drop the two stem indexes
        try:
            c.execute("DROP INDEX token_stems_stem ON token_stems")
        except MySQLdb._exceptions.OperationalError as e:
            if e.args[0] != MySQLdb_ER.CANT_DROP_FIELD_OR_KEY:
                raise
        try:
            c.execute("DROP INDEX token_stems_id ON token_stems")
        except MySQLdb._exceptions.OperationalError as e:
            if e.args[0] != MySQLdb_ER.CANT_DROP_FIELD_OR_KEY:
                raise

        # Delete all the existing stems from the table
        c.execute("DELETE FROM token_stems")

        self.commit()

    def update_token_stems(self, stemmer):
        # Stemmer is a CobeStemmer
        with trace_ms("Db.update_token_stems_ms"):
            c = self.cursor()

            insert_c = self.cursor()

            c.execute("SELECT id, text FROM tokens")
            q = c.fetchall()

            for row in q:
                stem = stemmer.stem(row[1])
                if stem is not None:
                    insert_c.execute("INSERT INTO token_stems (token_id, stem) VALUES (%s, %s)", (row[0], stem))

            self.commit()

        with trace_ms("Db.index_token_stems_ms"):
            try:
                c.execute("CREATE INDEX token_stems_id on token_stems (token_id)")
            except MySQLdb._exceptions.OperationalError as e:
                if e.args[0] != MySQLdb_ER.DUP_KEYNAME:
                    raise

            try:
                c.execute("CREATE INDEX token_stems_stem on token_stems (stem)")
            except MySQLdb._exceptions.OperationalError as e:
                if e.args[0] != MySQLdb_ER.DUP_KEYNAME:
                    raise

    def _run_migrations(self):
        with trace_us("Db.run_migrations_us"):
            self._maybe_create_node_count_triggers()

    def _maybe_create_node_count_triggers(self):
        # Create triggers on the edges table to update nodes counts.
        # In previous versions, the node counts were updated with a separate query.
        # Moving them into triggers improves performance.
        c = self.cursor()

        try:
            c.execute("CREATE TRIGGER edges_insert_trigger AFTER INSERT ON edges FOR EACH ROW UPDATE nodes SET count = count + NEW.count WHERE nodes.id = NEW.next_node")
        except MySQLdb._exceptions.OperationalError as e:
            if e.args[0] != MySQLdb_ER.TRG_ALREADY_EXISTS:
                raise

        try:
            c.execute("CREATE TRIGGER edges_update_trigger AFTER UPDATE ON edges FOR EACH ROW UPDATE nodes SET count = count + (NEW.count - OLD.count) WHERE nodes.id = NEW.next_node")
        except MySQLdb._exceptions.OperationalError as e:
            if e.args[0] != MySQLdb_ER.TRG_ALREADY_EXISTS:
                raise

        try:
            c.execute("CREATE TRIGGER edges_delete_trigger AFTER DELETE ON edges FOR EACH ROW UPDATE nodes SET count = count - old.count WHERE nodes.id = OLD.next_node")
        except MySQLdb._exceptions.OperationalError as e:
            if e.args[0] != MySQLdb_ER.TRG_ALREADY_EXISTS:
                raise
