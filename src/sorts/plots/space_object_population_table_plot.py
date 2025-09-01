from sorts.population import Population
import bokeh.models as bokeh_models


# TODO: re-eval if we need this
def space_object_population_table_plot(population: Population):
    source = bokeh_models.ColumnDataSource(data={f: population[f] for f in population.fields})
    columns = [bokeh_models.TableColumn(field=f, title=f) for f in population.fields]
    data_table = bokeh_models.DataTable(
        source=source,
        columns=columns,
        width=800,
        height=300,
        # selectable=True,
        # selectable="checkbox",
    )

    return data_table
