{{ name | escape | underline }}

.. automodule:: {{ fullname }}

.. currentmodule:: {{ fullname }}

Module summary
--------------

{% if attributes %}
.. rubric:: Attributes

.. autosummary::
   :toctree: .
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}

{% endif %}


{% if classes %}
.. rubric:: Classes

.. autosummary::
    :toctree: .
    {% for class in classes %}
    {{ class }}
    {% endfor %}

{% endif %}

{% if functions %}
.. rubric:: Functions

.. autosummary::
    :toctree: .
    {% for function in functions %}
    {{ function }}
    {% endfor %}

{% endif %}

Contents
----------

{% if attributes %}
.. rubric:: Attributes

{% for item in attributes %}

.. autoattribute:: {{ item }}
   :noindex:

{%- endfor %}
{% endif %}


{% if classes %}
{% for class in classes %}

.. rubric:: {{ class }}

.. autoclass:: {{ class }}
   :show-inheritance:
   :noindex:
   :members:
   :inherited-members:

{% endfor %}
{% endif %}

{% if functions %}
.. rubric:: Functions
{% for function in functions %}

.. autofunction:: {{ function }}
   :noindex:

{% endfor %}
{% endif %}
