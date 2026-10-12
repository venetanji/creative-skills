from __future__ import annotations
DYNAMIC_INPUT_LOOKUP = {"COMFY_AUTOGROW_V3"}
class DynamicPathsDefaultValue:
    EMPTY_DICT = "empty_dict"
def get_dynamic_input_func(kind):
    assert kind == "COMFY_AUTOGROW_V3"
    return expand_autogrow
def handle_prefix(prefix_list: list[str] | None, id: str | None=None) -> list[str]:
    if prefix_list is None:
        prefix_list = []
    if id is not None:
        prefix_list = prefix_list + [id]
    return prefix_list

def finalize_prefix(prefix_list: list[str] | None, id: str | None=None) -> str:
    assert not (prefix_list is None and id is None)
    if prefix_list is None:
        return id
    elif id is not None:
        prefix_list = prefix_list + [id]
    return '.'.join(prefix_list)

def expand_autogrow(out_dict: dict[str, Any], live_inputs: dict[str, Any], value: tuple[str, dict[str, Any]], input_type: str, curr_prefix: list[str] | None):
    is_names = 'names' in value[1]['template']
    is_prefix = 'prefix' in value[1]['template']
    input = value[1]['template']['input']
    if is_names:
        min = value[1]['template']['min']
        names = value[1]['template']['names']
        max = len(names)
    elif is_prefix:
        prefix = value[1]['template']['prefix']
        min = value[1]['template']['min']
        max = value[1]['template']['max']
        names = [f'{prefix}{i}' for i in range(max)]
    template_input = None
    template_required = True
    for _input_type, dict_input in input.items():
        if len(dict_input) == 0:
            continue
        template_input = list(dict_input.values())[0]
        template_required = _input_type == 'required'
        break
    if template_input is None:
        raise Exception('template_input could not be determined from required or optional; this should never happen.')
    new_dict = {}
    new_dict_added_to = False
    for i, name in enumerate(names):
        expected_id = finalize_prefix(curr_prefix, name)
        if i < min and template_required:
            out_dict['required'][expected_id] = template_input
            type_dict = new_dict.setdefault('required', {})
        else:
            out_dict['optional'][expected_id] = template_input
            type_dict = new_dict.setdefault('optional', {})
        if expected_id in live_inputs:
            type_dict[name] = template_input
            new_dict_added_to = True
    if not new_dict_added_to:
        finalized_prefix = finalize_prefix(curr_prefix)
        out_dict['dynamic_paths'][finalized_prefix] = finalized_prefix
        out_dict['dynamic_paths_default_value'][finalized_prefix] = DynamicPathsDefaultValue.EMPTY_DICT
    parse_class_inputs(out_dict, live_inputs, new_dict, curr_prefix)

def get_finalized_class_inputs(d: dict[str, Any], live_inputs: dict[str, Any], include_hidden=False) -> tuple[dict[str, Any], V3Data]:
    out_dict = {'required': {}, 'optional': {}, 'dynamic_paths': {}, 'dynamic_paths_default_value': {}}
    d = d.copy()
    hidden = d.pop('hidden', None)
    parse_class_inputs(out_dict, live_inputs, d)
    if hidden is not None and include_hidden:
        out_dict['hidden'] = hidden
    v3_data = {}
    dynamic_paths = out_dict.pop('dynamic_paths', None)
    if dynamic_paths is not None and len(dynamic_paths) > 0:
        v3_data['dynamic_paths'] = dynamic_paths
    dynamic_paths_default_value = out_dict.pop('dynamic_paths_default_value', None)
    if dynamic_paths_default_value is not None and len(dynamic_paths_default_value) > 0:
        v3_data['dynamic_paths_default_value'] = dynamic_paths_default_value
    return (out_dict, hidden, v3_data)

def parse_class_inputs(out_dict: dict[str, Any], live_inputs: dict[str, Any], curr_dict: dict[str, Any], curr_prefix: list[str] | None=None) -> None:
    for input_type, inner_d in curr_dict.items():
        for id, value in inner_d.items():
            io_type = value[0]
            if io_type in DYNAMIC_INPUT_LOOKUP:
                dynamic_input_func = get_dynamic_input_func(io_type)
                new_prefix = handle_prefix(curr_prefix, id)
                dynamic_input_func(out_dict, live_inputs, value, input_type, new_prefix)
            else:
                finalized_id = finalize_prefix(curr_prefix, id)
                out_dict[input_type][finalized_id] = value
                if curr_prefix:
                    out_dict['dynamic_paths'][finalized_id] = finalized_id

def build_nested_inputs(values: dict[str, Any], v3_data: V3Data):
    paths = v3_data.get('dynamic_paths', None)
    default_value_dict = v3_data.get('dynamic_paths_default_value', {})
    if paths is None:
        return values
    values = values.copy()
    result = {}
    create_tuple = v3_data.get('create_dynamic_tuple', False)
    for key, path in paths.items():
        parts = path.split('.')
        current = result
        for i, p in enumerate(parts):
            is_last = i == len(parts) - 1
            if is_last:
                value = values.pop(key, None)
                if value is None:
                    default_option = default_value_dict.get(key, None)
                    if default_option == DynamicPathsDefaultValue.EMPTY_DICT:
                        value = {}
                if create_tuple:
                    value = (value, key)
                current[p] = value
            else:
                current = current.setdefault(p, {})
    values.update(result)
    return values
