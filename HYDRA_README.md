The approach I took here with hydra is the early-validation approach using
ConfigStore pre-registration.  The following steps must be taken when writing any new
.yaml file

1. Define a new dataclass, inheriting it from any config group that it should be a
   part of.

For enum members that will be exposed to command-line overrides, use
`utils.MirroredEnum`.  This enables setting the enum value on the command-line.

```python
# in some file
@dataclass
class NewKindofDatasetOpts(DatasetOpts):
   ...
```

2. Add one line to register the schema name to the new dataclass.  The rule I'm
   following is that each schema corresponds to exactly one dataclass.  All overrides
   are provided as .yaml files.  Also, the name of the schema should end with
   `_schema`.  This ensures that the name won't conflict with the name of a yaml
   file, which is also searched as a potential schema name (confusing).  For example:

# in opts.py
```
def register_configs() -> None:
   ...
   # here, 'data' will correspond to the conf directory opts/data
   cs.store(group="data", name="new_kind_of_dataset_schema", node=NewKindOfDatasetOpts)
```

3. A new yaml file in `opts/data` for example, provides defaults for that particular
   schema, hence populates the dataclass.

There can be multiple yaml files all referencing the same schema.  Each one provides
a different set of defaults.

```yaml
defaults:
  - new_kind_of_dataset_schema@_here_
  - _self_
```

4. In your main program, you must call register_configs() and register_resolvers()
   before calling main.  For example:

```python
# expt/train_general.py (or expt/test_expr_dataset.py)
if __name__ == "__main__":
	register_configs()
	register_resolvers()
	main()
```

5. In the main function you will have:

```python
from omegaconf import DictConfig, OmegaConf
@hydra.main(config_path="./opts", config_name="some_yaml_stem", version_base="1.2")
def main(cfg: DictConfig):
	opts: MyTopLevelOpts = OmegaConf.to_object(cfg)
    ...
```

where there is a file `opts/some_yaml_stem.yaml` with the content:

```yaml
defaults:
  - my_top_level_schema@_here_
  - _self_
  ...
```

and where `my_top_level_schema` is registered to `MyTopLevelOpts` inside
register_configs().

