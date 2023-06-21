FROM grafana/xk6 as xk6_builder
RUN xk6 build --output k6 --with github.com/szkiba/xk6-faker@latest \
    --with github.com/grafana/xk6-output-prometheus-remote@latest 


FROM grafana/k6

WORKDIR /app/
COPY src/main.js /app/
COPY --from=xk6_builder /xk6/k6 /usr/bin/k6 

ENTRYPOINT k6 run -o xk6-prometheus-rw main.js