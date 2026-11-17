"""Declarative metadata only; importing this module creates no database objects."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
