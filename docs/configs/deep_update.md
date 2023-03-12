## Introduction ##

A method to merge two objects, named source and override, use override to modify source data.

## Usage ##

Source should be a dict or list, override must be a dict, otherwise return override directly.

### When source is a dict ###

#### delete key ####

Use `__delete__` keyword to delete all keys in source, `True` for delete all, `str` or `List[str]` for specific keys.

For example (using yaml to present python dict).

```yaml
# source
config:
    A:
        abc: 1
    B:
        a: d
        b: e
    C:
        A: a
        B: b
        C: c

# override
config:
    A:
        __delete__: true
    B:
        __delete__: b
    C:
        __delete__: [ A, B ]

# result
config:
    A: {}
    B:
        a: d
    C:
        C: c
```

#### add key and modify key ####

Just write the key-value pair to add or modify in source.

For example (using yaml to present python dict).

```yaml
# source
config:
    A:
        abc: 1
    B:
        a: d
        b: e

# override
config:
    A:
        abc: 2
    B:
        c: c
    C:
        a: A

# result
config:
    A:
        abc: 2
    B:
        a: d
        b: e
        c: c
    C:
        a: A
```

### When source is a list ###

#### delete item ####

Use `__delete__` keyword to delete all items in source, `True` for delete all, `int` or `List[int]` for specific items, negative int for counting from end.

For example (using yaml to present python dict).

```yaml
# source
config:
    A:
        - abc
        - efg
    B:
        - 123
        - 234
    C: [ a, b, c ]

# override
config:
    A:
        __delete__: true
    B:
        __delete__: 0
    C:
        __delete__: [ 0, -1 ]

# result
config:
    A: []
    B:
        - 234
    C:
        - b
```

#### modify item ####

Use `change_item` keyword to modify items in source, value mush be List[List[index, item]], change every index to corresponding item.

For example (using yaml to present python dict).

```yaml
# source
config:
    A:
        - abc
        - efg
    B: [ a, b, c ]

# override
config:
    A:
        change_item:
            - [ 0, A ]
    B:
        change_item:
            - [ -1, B ]
            - [ 0, C ]

# result
config:
    A:
        - A
        - efg
    B: [ C, b, B ]
```

#### add items ####

##### pre items #####

Use `pre_item` keyword to add items to the start of source, value mush be item or List[item]. If value if a list, add every item in it to source, otherwise, add the value to source.

For example (using yaml to present python dict).

```yaml
# source
config:
    A:
        - abc
        - efg
    B: [ a, b, c ]

# override
config:
    A:
        pre_item: A
    B:
        pre_item: [ B, C ]

# result
config:
    A:
        - A
        - abc
        - efg
    B:
        - B
        - C
        - a
        - b
        - c
```

##### post items #####

Use `post_item` keyword to add items to the end of source, value mush be item or List[item]. If value if a list, add every item in it to source, otherwise, add the value to source.

For example (using yaml to present python dict).

```yaml
# source
config:
    A:
        - abc
        - efg
    B: [ a, b, c ]

# override
config:
    A:
        post_item: A
    B:
        post_item: [ B, C ]

# result
config:
    A:
        - abc
        - efg
        - A
    B:
        - a
        - b
        - c
        - B
        - C
```

##### insert items #####

Use `insert_item` keyword to insert items to source, value mush be List[List[index, item] or List[index, item, extend]], insert item or a list of items to index position. If the index is bigger or equal than `len` of source or smaller than `-len` of source, this will work as `post_item` and `pre_item`. `extend` is a bool to indicated whether item is a list of items, you can omit it when it is False. This works well with `__delete__` and multi insert, the insert order and delete index will be calculated automatically.

For example (using yaml to present python dict).

```yaml
# source
config:
    A:
        - abc
        - efg
    B: [ a, b, c ]
    C: [ 1, 2, 3, 4 ]
    D: [ 1, 2, 3, 4 ]
    E: [ 1, 2, 3, 4 ]

# override
config:
    A:
        insert_item:
            - [ 0, A ]
            - [ 1, B ]
    B:
        insert_item:
            - [ -1, B ]
            - [ 1, [ 1, 2, 3 ], true ]
    C:
        insert_item:
            - [ -5, A ]
            - [ 4, B ]
            - [ 5, C ]
    D:
        __delete__: [ 1, 2 ]
        insert_item:
            - [ 0, A ]
            - [ 3, B ]
            - [ 1, [ C, D ], true ]
    E:
        __delete__: true
        insert_item:
            - [ 0, A ]
            - [ 3, B ]
            - [ 1, [ C, D ], true ]

# result
config:
    A:
        - A
        - abc
        - B
        - efg
    B:
        - a
        - 1
        - 2
        - 3
        - b
        - B
        - c
    C: [ A, 1, 2, 3, 4, B, C]
    D: [ A, 1, C, D, B, 4 ]
    E: [ A, C, D, B ]
```

## All keywords ##

| keyword       | value                                                                                                                                                      | effect                                                                                                 |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `__delete__`  | `True` or `str,int` or `list[str,int]`,`True` for delete all keys from other config, `str,int` only delete the specific key (for dict) or index (for list) | Delete some part of config from other.                                                                 |
| `change_item` | `list[[index, item]]`,used only when merge list                                                                                                            | Add ability of merg list, change the `list[index]` from other to `item`                                |
| `insert_item` | `list[[index, item, (extend)]]`,used only when merge list                                                                                                  | Add ability of merg list, insert iterm to the `list` at `index`, extend=True if insert a list of items |
| `pre_item`    | `Any`or `list[Any]`,used only when merge list                                                                                                              | Add ability of merg list, add the value in the start of the list from other to item                    |
| `post_item`   | `Any`or `list[Any]`,used only when merge list                                                                                                              | Add ability of merg list, add the value in the end of the list from other to item                      |

