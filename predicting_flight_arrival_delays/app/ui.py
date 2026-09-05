"""The interface, mounted over the API it consumes.

Two ways in, matching the two the API offers. Naming a flight - date, number, route,
the two carrier codes - is the one a person can actually use, so it comes first. The
other path wants every feature the model reads, which is a file
rather than a form: it is offered as a CSV upload, one row per flight.

Nothing here reaches into the service's own code. Every button goes through
app.wrapper, which sends the same HTTP requests an outside integrator would send.
"""

import gradio as gr

from predicting_flight_arrival_delays.app import wrapper

EXAMPLE = {
    "date": "2026-08-26",
    "marketing": "AA",
    "operating": "MQ",
    "number": 3500,
    "origin": "DFW",
    "dest": "LBB",
}


def build_lookup_tab() -> None:
    """The tab for scoring a flight the user only names."""
    gr.Markdown(
        "### Look up a flight\n"
        "Six fields, all of them printed on your booking.\n\n"
        "From those the service finds the flight in the timetable, then fills in the "
        "rest by itself: the scheduled times and the distance, how many other flights "
        "share each airport in that same hour, and the weather forecast for departure "
        "and for arrival."
    )

    with gr.Row():
        with gr.Column():
            flight_date = gr.DateTime(
                label="Departure date",
                include_time=False,
                type="string",
                value=EXAMPLE["date"],
            )
            number = gr.Number(
                label="Flight number", value=EXAMPLE["number"], precision=0, minimum=1
            )
        with gr.Column():
            origin = gr.Textbox(
                label="Departure airport", value=EXAMPLE["origin"], max_length=3, lines=1
            )
            dest = gr.Textbox(
                label="Arrival airport", value=EXAMPLE["dest"], max_length=3, lines=1
            )

    with gr.Row():
        marketing = gr.Textbox(
            label="Airline selling the flight",
            value=EXAMPLE["marketing"],
            max_length=3,
            lines=1,
            info="The code printed on the ticket. It is what finds the flight.",
        )
        operating = gr.Textbox(
            label="Airline operating the flight",
            value=EXAMPLE["operating"],
            max_length=3,
            lines=1,
            info=(
                "Often the same one. On regional flights the ticket says "
                "'operated by ...': that code is the one the model was trained on."
            ),
        )

    predict = gr.Button("Estimate the risk of delay", variant="primary")
    answer = gr.Markdown()

    contributions = gr.Plot(show_label=False)

    predict.click(
        fn=wrapper.predict_lookup,
        inputs=[flight_date, marketing, operating, number, origin, dest],
        outputs=[answer, contributions],
    )


def build_batch_tab() -> None:
    """The tab for scoring flights already described in full."""
    gr.Markdown(
        "### A group of flights\n"
        "For callers who already hold the data: a CSV with one row per flight, "
        "carrying the columns the served models read."
    )

    uploaded = gr.File(label="CSV of flights", file_types=[".csv"])
    score = gr.Button("Score them all", variant="primary")
    results = gr.Dataframe(label="Results")

    score.click(fn=wrapper.predict_batch, inputs=uploaded, outputs=results)


def build() -> gr.Blocks:
    """Assemble the interface.

    Returns:
        The Blocks app, ready to be mounted on the API.
    """
    with gr.Blocks(title="FlightOnTime") as interface:
        gr.Markdown(
            "# ✈️ FlightOnTime\nArrival delay risk for scheduled U.S. flights, before departure."
        )

        with gr.Tabs():
            with gr.TabItem("Look up a flight"):
                build_lookup_tab()

            with gr.TabItem("Group of flights"):
                build_batch_tab()

            with gr.TabItem("Metrics"):
                gr.Markdown("How the served models scored when they were released.")
                metrics = gr.Markdown()
                interface.load(fn=wrapper.get_metrics, outputs=metrics)

            with gr.TabItem("Hyperparameters"):
                hyperparameters = gr.Markdown()
                interface.load(fn=wrapper.get_hyperparameters, outputs=hyperparameters)

            with gr.TabItem("What to send"):
                inputs = gr.Markdown()
                interface.load(fn=wrapper.get_inputs, outputs=inputs)

    return interface
