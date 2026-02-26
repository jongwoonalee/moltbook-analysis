import sqlite3

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)

    return conn

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except sqlite3.Error as e:
        print(e)

def main():
    database = "moltbook.db"

    sql_create_posts_table = """ CREATE TABLE IF NOT EXISTS posts (
                                        id TEXT PRIMARY KEY,
                                        title TEXT NOT NULL,
                                        content TEXT,
                                        upvotes INTEGER,
                                        downvotes INTEGER,
                                        comment_count INTEGER,
                                        created_at TEXT,
                                        FOREIGN KEY (submolt_id) REFERENCES submolts (id),
                                        submolt_name TEXT,
                                        author_id TEXT,
                                        author_name TEXT
                                    ); """

    sql_create_comments_table = """CREATE TABLE IF NOT EXISTS comments (
                                    id TEXT PRIMARY KEY,
                                    post_id TEXT NOT NULL,
                                    content TEXT,
                                    upvotes INTEGER,
                                    downvotes INTEGER,
                                    created_at TEXT,
                                    author_id TEXT,
                                    author_name TEXT,
                                    parent_comment_id TEXT,
                                    FOREIGN KEY (post_id) REFERENCES posts (id)
                                );"""

    sql_create_submolts_table = """CREATE TABLE IF NOT EXISTS submolts (
                                    id TEXT PRIMARY KEY,
                                    name TEXT,
                                    created_at TEXT,
                                    subscriber_count INTEGER
                                );"""

    # create a database connection
    conn = create_connection(database)

    # create tables
    if conn is not None:
        create_table(conn, sql_create_posts_table)
        create_table(conn, sql_create_comments_table)
        create_table(conn, sql_create_submolts_table)
        conn.close()
    else:
        print("Error! cannot create the database connection.")

if __name__ == '__main__':
    main()
