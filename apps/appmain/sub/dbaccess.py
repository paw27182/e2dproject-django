import datetime
import logging
import sqlite3
from datetime import datetime as dt
from pathlib import Path

from django.conf import settings

from .dbabstractclass import DBAbstractClass

mid = Path(__file__).name
ftime = "%Y/%m/%d %H:%M:%S"


class DBAccess(DBAbstractClass):
    def __init__(self, dbname, tablename, /, key=None, key_type=None, primary_key=None):
        super().__init__(dbname, tablename)

        BASE_DIR = settings.BASE_DIR

        self.connect_str = Path(BASE_DIR, "database/{}.sqlite3".format(dbname))
        self.tablename = tablename

        if key:  # in case create table
            key += ("update_at", "modified_by")
            key_type += ("text", "text")
            col = ", ".join([key[i] + " " + key_type[i] for i in range(len(key))])

            # generate CREATE table statement
            if primary_key:
                pk = ", ".join(primary_key)
                sql = f"CREATE TABLE IF NOT EXISTS '{tablename}' ({col}, PRIMARY KEY({pk}));"
            else:
                sql = f"CREATE TABLE IF NOT EXISTS '{tablename}' ({col});"
            # print(f"[{mid}] {sql= }")
            self.key = key
            try:
                with sqlite3.connect(self.connect_str) as con:
                    con.execute(sql)  # create a table
            except Exception as err:
                print(f"[{mid}] Database initialization error occurred.  {err= }")
        else:
            sql = f"SELECT * FROM SQLITE_MASTER WHERE TYPE='table' AND NAME='{tablename}';"

            with sqlite3.connect(self.connect_str) as con:
                cur = con.execute(sql)
                s = cur.fetchall()

            s = s[0][4]  # extract CREATE Database statement

            if "PRIMARY KEY" in s:
                s = s[s.index("(") + 1:s.index("PRIMARY KEY") - 2].split(",")  # '(' ～ 'modified_by'
            else:
                s = s[s.index("(") + 1:-6].split(",")  # '(' ～ 'modified_by'
            s = [x.lstrip() for x in s]
            self.key = [x.split(" ")[0] for x in s]

    def get_form_size(self, selected_form=None):
        with sqlite3.connect(self.connect_str) as con:
            c = con.execute(f"SELECT count(*) FROM '{selected_form}';")   # quote
            form_size = c.fetchone()  # tuple
        return form_size[0]

    # def get_table_names(self):
    #     # find table names
    #     with sqlite3.connect(self.connect_str) as con:
    #         sql = 'SELECT NAME FROM SQLITE_MASTER WHERE TYPE="table";'
    #         cur = con.execute(sql)
    #         result = cur.fetchall()
    #     table_names = [item[0] for item in result]
    #     return table_names

    def write_form_items(self, input_data, current_username):
        # generate insert or update statement
        questions = ", ".join(list(('?' * len(self.key))))
        col = ", ".join(self.key)
        sql = f"REPLACE INTO '{self.tablename}' ({col}) VALUES({questions});"
        # print(f"[{mid}]{sql= }")

        update_at = dt.now().strftime(ftime)
        modified_by = current_username
        try:
            with sqlite3.connect(self.connect_str) as con:
                chunk_size = 1000

                if len(input_data) < chunk_size:
                    for row in input_data:  # input_data: list [[],[], ...]
                        value = row.copy()
                        value += [update_at, modified_by]
                        con.execute(sql, value)
                        # con.commit()  # unnecessary because isolation_level = None -> Auto commit mode
                else:
                    for i in range(int(len(input_data) / chunk_size) + 1):
                        chunk = input_data[i * chunk_size: (i + 1) * chunk_size]
                        values = list()
                        for row in chunk:
                            value = row.copy()
                            value += [update_at, modified_by]
                            values.append(value)
                        con.executemany(sql, values)
            return "OK", "Done."  # alert

        except Exception as err:
            return "NG", f"error: [{mid}] Database access error occurred. {err= } {sql= } {value= }"  # alert

    def read_form_items(self, command):
        print(f"[{mid}] find_items {command= }")

        order_by = self.key[0] + " ASC, update_at"  # sort by the first column and update_at
        col = ",".join([self.key[i] for i in range(len(self.key))])

        if "show_the_first_item" in command:
            sql = f"SELECT {col} FROM '{self.tablename}'  ORDER BY {order_by} ASC LIMIT 1;"

        elif "show_items_updated_today" in command:
            today = dt.now().strftime("%Y/%m/%d")
            sql = f"SELECT {col} FROM '{self.tablename}' WHERE update_at LIKE '{today}%' ORDER BY {order_by} ASC;"

        elif "show_items_updated_in_a_week" in command:
            week = (dt.now() - datetime.timedelta(days=7)).strftime("%Y/%m/%d")
            sql = f"SELECT {col} FROM '{self.tablename}' WHERE update_at >= '{week}' ORDER BY {order_by} ASC;"

        else:  # "show_all_the_items" in command:
            sql = f"SELECT {col} FROM '{self.tablename}' ORDER BY {order_by} ASC;"

        # print(f"[{mid}] {sql= }")

        with sqlite3.connect(self.connect_str) as conn:
            cur = conn.execute(sql)
            result = cur.fetchall()

        items = list()

        for item in result:
            items.append(list(item))  # tuple to list

        return self.key, items

    def drop_form(self):
        with sqlite3.connect(self.connect_str) as con:
            sql = f"DROP TABLE '{self.tablename}';"
            con.execute(sql)

        con = sqlite3.connect(self.connect_str)
        sql = 'select name from sqlite_master where type="table";'
        cur = con.execute(sql)
        tablenames = cur.fetchall()
        con.close()  # in case delete a file

        if not len(tablenames):
            try:
                Path(self.connect_str).unlink()  # delete .sqlite3 file
            except Exception as err:
                print(f"[{mid}] {err= }")

    def delete_item(self, record_to_be_processed):
        s = record_to_be_processed
        s = s.replace(", ", ",")  # trimming
        items = s[1:-1].split(",")  # remove '(' and ')' and split
        items = items[1:]  # remove the first number column

        condition = ""
        for k, v in zip(self.key, items):
            if v != "None":
                condition += k + "=" + v + " AND "

        condition = condition[:-5]  # remove the last " AND"

        with sqlite3.connect(self.connect_str) as con:
            sql = f"DELETE FROM '{self.tablename}' WHERE {condition};"
            con.execute(sql)
