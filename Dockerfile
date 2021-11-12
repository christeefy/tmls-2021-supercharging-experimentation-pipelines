FROM python:3.8-slim-buster

WORKDIR /opt

RUN pip install -U pip==21.2.4 \
    && pip install poetry==1.1.11


COPY pyproject.toml poetry.lock /opt/
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction

COPY lightning_entrypoint.py predict_script.py /opt/
COPY src/ /opt/src/

# ENTRYPOINT [ "python", "-m", "predict_script" ]
ENTRYPOINT ["/bin/bash"]
