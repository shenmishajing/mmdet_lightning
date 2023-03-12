## Introduction ##

The default parser for lightning CLI, a yaml parser with [deep update](../deep_update.md) feature.

## Usage ##

Use `--config` flag of lightning CLI to load a yaml config file with this parser. Support `__base__` and all [deep update](../deep_update.md) keyword, for detail of [deep update](../deep_update.md), see [deep update](../deep_update.md).

## Config inherit ##

Use `__base__` keyword to inherit config files, value must be `str` or `List[str]`, which is relativa path of config file to inherit, can appear at any dict level.

For example (using yaml to present python dict).

```yaml
# configs/model/example.yaml
config:
    A:
        abc: 1
    B:
        a: d
    C:
        A: a
        C: c

# configs/model/test.yaml
config:
    A:
        abc: 1
    B:
        b: e
    C:
        B: b
        C: d

# configs/runs/example.yaml
__base__: ../model/example.yaml
config:
    B:
        c: b
    C:
        __base__: [ [../model/test.yaml, config.C ] ]
        D: f
    D:
        __base__: [ [../model/test.yaml, config.C ] ]
        C: f

# result
config:
    A:
        abc: 1
    B:
        a: d
        c: b
    C:
        A: a
        B: b
        C: d
    D:
        B: b
        C: f
```
## Yaml import ##

Yaml support anchor and aliased feature, but it's not convenient sometimes, so, you can use `__import__` keyword to create the anchor first, and use them in the config file. Carefully, the `__import__` keyword is only supported at the top level of config file.

For example (using yaml to present python dict).

```yaml
__import__:
    # all config under __import__ will be ignored
    # this part is only used for creating anchors
    import_train_img_scale_kwargs: &import_train_img_scale_kwargs
        img_scale: !!python/tuple [ 512, 512 ]
        keep_ratio: false
    import_test_img_scale_kwargs: &import_test_img_scale_kwargs
        img_scale: !!python/tuple [ 128, 128 ]
        keep_ratio: true

dataset_cfg:
    pipeline:
        train:
            -   type: Resize
                <<: *import_train_img_scale_kwargs
        test:
            -   type: Resize
                <<: *import_test_img_scale_kwargs
```

```yaml

dataset_cfg:
    __import__:
        # unsuported __import__ keyword not at top level!
        # will not be ignored!
        import_train_img_scale_kwargs: &import_train_img_scale_kwargs
            img_scale: !!python/tuple [ 512, 512 ]
            keep_ratio: false
        import_test_img_scale_kwargs: &import_test_img_scale_kwargs
            img_scale: !!python/tuple [ 128, 128 ]
            keep_ratio: true
    pipeline:
        train:
            -   type: Resize
                <<: *import_train_img_scale_kwargs
        test:
            -   type: Resize
                <<: *import_test_img_scale_kwargs
```

## All keywords ##

| keyword       | value                                                                                                                                                                                            | effect                                                                                                 |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| `__base__`    | `str` or `list[str]` or `list[[str, str]]`,(each `str` should be a relative path from current cofig file, when there are two str, the second one will be the key (using `.` to split) to import) | Merge every config one by one, current last.                                                           |
| `__import__`  | Any                                                                                                                                                                                              | Just delete this, for convenience of reference in yaml                                                 |
| `__delete__`  | `True` or `str,int` or `list[str,int]`,`True` for delete all keys from other config, `str,int` only delete the specific key (for dict) or index (for list)                                       | Delete some part of config from other.                                                                 |
| `change_item` | `list[[index, item]]`,used only when merge list                                                                                                                                                  | Add ability of merg list, change the `list[index]` from other to `item`                                |
| `insert_item` | `list[[index, item, (extend)]]`,used only when merge list                                                                                                                                        | Add ability of merg list, insert iterm to the `list` at `index`, extend=True if insert a list of items |
| `pre_item`    | `Any`or `list[Any]`,used only when merge list                                                                                                                                                    | Add ability of merg list, add the value in the start of the list from other to item                    |
| `post_item`   | `Any`or `list[Any]`,used only when merge list                                                                                                                                                    | Add ability of merg list, add the value in the end of the list from other to item                      |
