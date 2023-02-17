from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode, JsCode, DataReturnMode

MAX_TABLE_HEIGHT = 800


def get_numeric_style_with_precision(precision: int) -> dict:
    return {"type": ["numericColumn", "customNumericFormat"], "precision": precision}


PRECISION_ZERO = get_numeric_style_with_precision(0)
PRECISION_ONE = get_numeric_style_with_precision(1)
PRECISION_TWO = get_numeric_style_with_precision(2)
PINLEFT = {"pinned": "left"}


def draw_grid(
        df,
        formatter: dict = None,
        selection="multiple",
        use_checkbox=False,
        pre_selected_rows: list = None,
        fit_columns=False,
        theme="streamlit",
        max_height: int = MAX_TABLE_HEIGHT,
        wrap_text: bool = True,
        auto_height: bool = False,
        grid_options: dict = None,
        key=None,
        css: dict = None
):

    gb = GridOptionsBuilder.from_dataframe(df, formatter=formatter)
    gb.configure_default_column(
        filterable=True,
        groupable=False,
        editable=False,
        wrapText=wrap_text,
        autoHeight=auto_height,
        data_return_mode=DataReturnMode.AS_INPUT,
        wrapHeaderText=True
    )

    if grid_options is not None:
        gb.configure_grid_options(**grid_options)

    for latin_name, (cyr_name, style_dict) in formatter.items():
        gb.configure_column(latin_name, header_name=cyr_name, **style_dict)

    gb.configure_selection(selection_mode=selection, use_checkbox=use_checkbox, pre_selected_rows=pre_selected_rows)

    return AgGrid(
        df,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=fit_columns,
        height=min(max_height, (1 + len(df.index)) * 30),
        theme=theme,
        key=key,
        custom_css=css
    )

def draw_grid2(
        df,
        formatter: dict = None,
        selection="multiple",
        use_checkbox=False,
        pre_selected_rows: list = None,
        fit_columns=False,
        theme="streamlit",
        max_height: int = MAX_TABLE_HEIGHT,
        wrap_text: bool = True,
        auto_height: bool = False,
        grid_options: dict = None,
        key=None,
        css: dict = None
):

    gb = GridOptionsBuilder()
    gb.configure_default_column(
        filterable=True,
        groupable=False,
        editable=False,
        wrapText=wrap_text,
        autoHeight=auto_height,
        wrapHeaderText=True,
    )

    if grid_options is not None:
        gb.configure_grid_options(**grid_options)

    for latin_name, (cyr_name, style_dict) in formatter.items():
        gb.configure_column(latin_name, header_name=cyr_name, **style_dict)

    gb.configure_selection(selection_mode=selection, use_checkbox=use_checkbox, pre_selected_rows=pre_selected_rows)

    return AgGrid(
        df,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=fit_columns,
        height=460,
        theme=theme,
        key=key,
        custom_css=css
    )

def highlight(color, condition):
    code = f"""
        function(params) {{
            color = "{color}";
            if ({condition}) {{
                return {{
                    'backgroundColor': color
                }}
            }}
        }};
    """
    return JsCode(code)
