import sqlite3,click
from flask import current_app
from flask import g
from flask.cli import with_appcontext


def get_db():
    """Connect to the application"s configured database. The connection
    is unique for each request and will be reused if this is called
    again.
    """
    if "db" not in g:
        g.db=sqlite3.connect(
            current_app.config["DATABASE"],detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory=sqlite3.Row
    return g.db


def close_db(e=None):
    """If this request connected to the database,close the
    connection.
    """
    db=g.pop("db",None)
    if db is not None:
        db.close()


@click.command("init-db")
@with_appcontext
def init_db_command():
    """Clear existing data and create new tables."""
    db=get_db()
    with current_app.open_resource("schema.sql") as f:
        db.executescript(f.read().decode("utf-8"))
    click.echo("Initialized the database.")


def init_app(app):
    """Register database functions with the Flask beaker. This is called by
    the application factory.
    """
    #告诉Flask在返回响应后进行清理的时候调用此函数
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
