#!/usr/bin/env python3

from copy import copy

import numpy as np
import awkward as ak


class LazyArray:
    """"
    Wrapper around awkward arrays that lazily constructs fields only when necessary.
    This is meant for cases where large arrays are read, but then downselected to much smaller arrays.
    Calculating all derived fields before this selection step, which might required only some of them, is inefficient.
    """
    def __init__(self, awkward_array, known_fields, lazy_fields):
        self.awkward_array = awkward_array
        self.known_fields = known_fields
        self.lazy_fields = lazy_fields

    @staticmethod
    def create(awkward_array):
        known_fields = set(awkward_array.type.content.fields)
        return LazyArray(awkward_array, known_fields, {})

    def add_field(self, name, producer):
        if name in self.known_fields:
            raise ValueError(f"Field {name!r} is already present.")
        elif name in self.lazy_fields:
            print(f"Warning: Overwriting field {name!r}")
        self.lazy_fields[name] = producer

    def with_field(self, name, values):
        self.known_fields.add(name)
        self.awkward_array = ak.with_field(self.awkward_array, values, name)

    def get_field(self, field_name):
        if field_name in self.known_fields:
            # it's already present in awkward_array
            return self.awkward_array[field_name]
        elif field_name in self.lazy_fields:
            # calculate the values now
            field_value = self.lazy_fields[field_name](self)
            if field_value is None:
                raise ValueError(f"Function calculating {field_name!r} returned None.")
            self.awkward_array = ak.with_field(self.awkward_array, field_value, field_name)
            self.known_fields.add(field_name)
            self.lazy_fields.pop(field_name)
            return field_value

    def __getitem__(self, key):
        if not isinstance(key, str):
            # some kind of indexing
            selected_array = self.awkward_array[key]
            return LazyArray(selected_array, copy(self.known_fields), copy(self.lazy_fields))
        # Selecting a field
        result = self.get_field(key)
        if result is None:
            raise ValueError(f"Field {key!r} is unknown.")
        return result

    def __getattr__(self, key):
        value = self.get_field(key)
        if value is None:
            raise AttributeError
        return value

    def __len__(self):
        return len(self.awkward_array)

    def get_fields(self):
        return sorted(set(self.known_fields) | set(self.lazy_fields))

    def get_dtypes(self):
        return dict(zip(self.awkward_array.type.content.fields, self.awkward_array.type.content.contents))

    def get_array(self):
        return ak.Array({field: self[field] for field in self.get_fields()})
