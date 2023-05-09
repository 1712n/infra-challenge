    import http from 'k6/http';
    import { check } from 'k6';
    import { Faker } from "k6/x/faker"

    let f = new Faker();

    export const options = {
        // Tag for grafana dashboard
        tags: {
            testid: `${__ENV.PARTICIPANT_NAME}`,
        },
        scenarios: {
            does_it_even_work: {
                executor: 'constant-arrival-rate',
                rate: 1,
                timeUnit: '1s',
                duration: '1m',
                preAllocatedVUs: 1,
                maxVUs: 1,
                exec: 'post_radom',
            },

            highload_scenario: {
                executor: 'ramping-arrival-rate',
                startTime: '61s',
                preAllocatedVUs: 10,
                maxVUs: 50,
                timeUnit: '1s',
                startRate: 1,
                stages: [
                    { target: 10, duration: '5s' },
                    { target: 30, duration: '5s' },
                    { target: 60, duration: '10s' },
                    { target: 60, duration: '1m' },
                    { target: 40, duration: '1m' },
                    { target: 60, duration: '10s' },
                    { target: 10, duration: '30s' },
                    { target: 0, duration: '5s' },
                ],
                exec: 'post_radom',
            },

            spike_scenario: {
                executor: 'ramping-arrival-rate',
                startTime: '247s',
                preAllocatedVUs: 10,
                maxVUs: 50,
                timeUnit: '1s',
                startRate: 1,
                stages: [
                    { target: 10, duration: '5s' },
                    { target: 30, duration: '5s' },
                    { target: 200, duration: '10s' },
                    { target: 200, duration: '2m' },
                    { target: 0, duration: '5s' },
                ],
                exec: 'post_radom',
            },

            stress_speed_test: {
                executor: 'shared-iterations',
                startTime: '392s',
                vus: 10,
                iterations: 1000000,
                maxDuration: '2m',
                exec: 'post_radom',
            },

        }
    }


    export function post_radom() {
        let text = ""
        // 0.5 percent of requests will be empty
        if (Math.random() < 0.005) {
            text = ""
        } else {
            text = f.hipsterSentence(f.randomInt([0, 10])) + f.emoji() + f.url() + f.hipsterSentence(f.randomInt([0, 10])) + ". " + f.regex("[\x00-\x7F]{5,20}") + " " + f.bitcoinAddress() + f.emoji() + " " + f.hackerPhrase(f.randomInt([0, 10]))
        }

        console.log("mytext: " + text)

        // post json to the API
        const res = http.post('http://localhost:8000/process', { text: text }, {
            headers: {
                'Content-Type': 'application/json',
                'accept': 'application/json',
            },
        });
        const languages_list = ["Arabic", "Basque", "Breton", "Catalan", "Chinese_China", "Chinese_Hongkong", "Chinese_Taiwan", "Chuvash", "Czech", "Dhivehi", "Dutch", "English", "Esperanto", "Estonian", "French", "Frisian", "Georgian", "German", "Greek", "Hakha_Chin", "Indonesian", "Interlingua", "Italian", "Japanese", "Kabyle", "Kinyarwanda", "Kyrgyz", "Latvian", "Maltese", "Mongolian", "Persian", "Polish", "Portuguese", "Romanian", "Romansh_Sursilvan", "Russian", "Sakha", "Slovenian", "Spanish", "Swedish", "Tamil", "Tatar", "Turkish", "Ukranian", "Welsh"]


        check(res, {
            'is status 200': (r) => r.status === 200,
            "model 1 score valid": res => res.json().cardiffnlp.score > 0,
            "model 1 label valid": res => res.json().cardiffnlp.label === "POSITIVE" || res.json().cardiffnlp.label === "NEGATIVE" || res.json().cardiffnlp.label === "NEUTRAL",

            "model 2 score valid": res => res.json().ivanlau.score > 0,
            "model 2 label valid": res => languages_list.includes(res.json().ivanlau.label),

            "model 3 score valid": res => res.json().svalabs.score > 0,
            "model 3 label valid": res => res.json().svalabs.label === "SPAM" || res.json().svalabs.label === "HAM",

            "model 4 score valid": res => res.json().EIStakovskii.score > 0,
            "model 4 label valid": res => res.json().EIStakovskii.label === "LABEL_0" || res.json().EIStakovskii.label === "LABEL_1",

            "model 5 score valid": res => res.json().jy46604790.score > 0,
            "model 5 label valid": res => res.json().jy46604790.label === "LABEL_0" || res.json().jy46604790.label === "LABEL_1",
        });
    }
