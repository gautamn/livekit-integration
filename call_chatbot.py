import requests
import json


def get_chat_messages(conversation_id: str, token: str):
    url = f'https://app.eng.quant.ai/console/api/apps/cdf4e5b4-7411-4362-a073-1af8046a8c9b/chat-messages?conversation_id={conversation_id}'

    headers = {
        'Authorization': f'Bearer {token}'
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Request failed with status {response.status_code}: {response.text}")


def query_chatbot(conversation_id: str, bearer_token: str, query: str):
    url = "https://app.eng.quant.ai/console/api/apps/cdf4e5b4-7411-4362-a073-1af8046a8c9b/chat-messages"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {bearer_token}"
    }

    data = {
            "response_mode": "streaming",
            "conversation_id": conversation_id,
            "query": query,
            "inputs": {},
            "model_config": {
                "external_data_tools": [],
                "pre_prompt": "You are a pizza ordering agent specializing in combo orders, guiding users step-by-step through selecting a location, building their pizza, wings, and sodas, and completing their order with clear confirmations at every stage.\n\n1. Provide Combo Info First (if prompted to):\nExplain that the combo includes a pizza, wings, and 4 drinks.\nBase price (small, medium, or large): $32.59 before tax.\nX-Large pizza is $4 extra.\nPremium toppings, sauces, and drinks may add extra cost.\nAnswer any user questions before starting.\n\n2. Location Selection:\nAsk the user for their location after they agree to start.\nUse LocationNameToLatsLongs to find 3 nearby store options.\nNever assume or guess location.\nConfirm which store the user selects (retrieve store_id).\n\n3. Cart Initialization:\nAfter store is chosen, use POST Init Cart with the correct store_id.\n\n4. Building the Order:\nGuide user step-by-step through the combo build:\nselectPizzaSize, selectCrustSauceCheese, selectPizzaToppings (standard toppings), addExtraToppings (premium toppings), addSpecialInstructions (e.g., \"well done\"), selectWingType (classic or breaded), selectWingSauce, selectSodas.\nAlways confirm choices after each tool is used (\"I have added your crust and sauce,\" etc.)\n\n5. Final Review and Completing Order:\nOnce user says order is done, use finalOrderAssembly to summarize the order and add to cart.\nMust use finalOrderAssembly before using completeOrder to officially submit the order.\n\n6. Cart Edits (If Needed):\nOnly use editCart if the user asks to change something.\nThis tool can only be used after the finalOrderAssembly tool was used.\n\n7. Order Completion:\nOnce the order is complete and user inquires about when they can pick up their food, use the POST Future Store Hours tool to have them choose a time.\n\nRules:\nNo cart initialization until store is picked.\nNo order building until cart is initialized.\nAlways confirm each step and action clearly.\nStay in control but give the user the lead — only proceed when user agrees.\nDo not place order without getting user information.\nYou MUST let user choose a pickup time.\nMake sure your responses are extremely extremely short and to the point.",
                "postprocessing_prompt": [
                    {
                        "channel": "web",
                        "prompt": "Your responses should be humble and polite and in a professional manner. When possible, provide information in a tabular form and using markdown formatting."
                    },
                    {
                        "channel": "voice",
                        "prompt": "Your responses should be crisp, humble, polite and in a professional manner. Never include any tabular, new-line characters or markdown formatting or html formatting, asterisk in the response also do not have any numbered list and also do not have characters like *,#.\\,>,<,-,+ etc.... Ensure your responses are short and can be read out by an agent in a phone call but ensure to read out the critical information in a summarized form. Don't include any formulas or equations and just try summarizing or giving the final answer in short. Use ssml tags where ever possible to read out phone numbers, account numbers, dates. Please avoid using expressions like ‘greater than’ or ‘less than’ in the call—describe amounts logically instead. Also, say ‘US Dollars’ instead of saying ‘United States Dollars'. Use correct punctuations and take pauses wherever necessary."
                    }
                ],
                "use_postprocessing_prompt": True,
                "prompt_type": "simple",
                "chat_prompt_config": {},
                "completion_prompt_config": {},
                "user_input_form": [],
                "dataset_query_variable": "",
                "opening_statement": "PIZZA TIME! What you got for me?",
                "more_like_this": {
                    "enabled": False
                },
                "suggested_questions": [],
                "suggested_questions_after_answer": {
                    "enabled": False
                },
                "text_to_speech": {
                    "enabled": False,
                    "voice": "",
                    "language": ""
                },
                "speech_to_text": {
                    "enabled": False
                },
                "retriever_resource": {
                    "enabled": True
                },
                "sensitive_word_avoidance": {
                    "enabled": False,
                    "type": "",
                    "configs": []
                },
                "agent_mode": {
                    "max_iteration": 10,
                    "enabled": True,
                    "strategy": "function_call",
                    "tools": [
                        {
                            "provider_id": "a385172d-c409-4934-85d5-37085aefbdb8",
                            "provider_type": "workflow",
                            "provider_name": "POST Init Cart",
                            "tool_name": "PostInitCart",
                            "tool_label": "POST Init Cart",
                            "tool_parameters": {
                                "order_type": "",
                                "store_id": ""
                            },
                            "enabled": True,
                            "isDeleted": False,
                            "notAuthor": False
                        },
                        {
                            "provider_id": "bbaa597d-6eb3-433d-afda-be91ba2f0897",
                            "provider_type": "workflow",
                            "provider_name": "GET Pickup Store Search",
                            "tool_name": "GetPickupStoreSearch",
                            "tool_label": "GET Pickup Store Search",
                            "tool_parameters": {
                                "latitude": "",
                                "longitude": ""
                            },
                            "enabled": False,
                            "isDeleted": False,
                            "notAuthor": False
                        },
                        {
                            "provider_id": "5fb26fae-c36e-452f-a489-396467ddd0ce",
                            "provider_type": "workflow",
                            "provider_name": "GET Product Config",
                            "tool_name": "GetProductConfig",
                            "tool_label": "GET Product Config",
                            "tool_parameters": {
                                "store_id": "",
                                "product_slug": "",
                                "product_id": ""
                            },
                            "enabled": True,
                            "isDeleted": False,
                            "notAuthor": False
                        },
                        {
                            "provider_id": "2e958332-09f3-488d-8e90-178fdd4027ad",
                            "provider_type": "workflow",
                            "provider_name": "GET Product List",
                            "tool_name": "GetProductList",
                            "tool_label": "GET Product List",
                            "tool_parameters": {
                                "store_id": "",
                                "delivery_type": "",
                                "category_id": ""
                            },
                            "enabled": True,
                            "isDeleted": False,
                            "notAuthor": False
                        },
                        {
                            "provider_id": "172334be-0edd-4a03-8d6d-fa7c047bf4cd",
                            "provider_type": "workflow",
                            "provider_name": "GET Category List",
                            "tool_name": "GetCategoryList",
                            "tool_label": "GET Category List",
                            "tool_parameters": {
                                "store_id": ""
                            },
                            "enabled": True,
                            "isDeleted": False,
                            "notAuthor": False
                        },
                        {
                            "provider_id": "aee0d611-42cc-4a8d-bfa1-683de1a4734a",
                            "provider_type": "workflow",
                            "provider_name": "selectPizzaSize",
                            "tool_name": "selectPizzaSize",
                            "tool_label": "selectPizzaSize",
                            "tool_parameters": {
                                "pizzaSize": ""
                            },
                            "enabled": True,
                            "isDeleted": False,
                            "notAuthor": False
                        },
                        {
                            "provider_id": "b8b845e6-f2f3-4a3d-8db7-d9d66687ffd1",
                            "provider_type": "workflow",
                            "provider_name": "selectCrustSauceCheese",
                            "tool_name": "selectCrustSauceCheese",
                            "tool_label": "selectCrustSauceCheese",
                            "tool_parameters": {
                                "crust": "",
                                "sauce": "",
                                "cheese": "",
                                "pizzaSize": ""
                            },
                            "enabled": True,
                            "isDeleted": False,
                            "notAuthor": False
                        },
                        {
                            "provider_id": "e7764373-94e8-41ad-820f-482e48acfed0",
                            "provider_type": "workflow",
                            "provider_name": "LocationNameToLatsLongs",
                            "tool_name": "LocationNameToLatsLongs",
                            "tool_label": "LocationNameToLatsLongs",
                            "tool_parameters": {
                                "location_name": ""
                            },
                            "enabled": True,
                            "isDeleted": False,
                            "notAuthor": False
                        },
                        {
                            "provider_id": "44078863-d2ab-4e4c-b670-1d295ae60d48",
                            "provider_type": "workflow",
                            "provider_name": "selectPizzaToppings",
                            "tool_name": "selectPizzaToppings",
                            "tool_label": "selectPizzaToppings",
                            "tool_parameters": {
                                "topping": "",
                                "quantityOfTopping": "",
                                "direction": ""
                            },
                            "enabled": True,
                            "isDeleted": False,
                            "notAuthor": False
                        },
                        {
                            "provider_id": "fb8ae190-daa0-46b7-b169-1de069afc447",
                            "provider_type": "workflow",
                            "provider_name": "addExtraToppings",
                            "tool_name": "addExtraToppings",
                            "tool_label": "addExtraToppings",
                            "tool_parameters": {
                                "topping": "",
                                "quantityOfTopping": "",
                                "direction": ""
                            },
                            "enabled": True,
                            "isDeleted": False,
                            "notAuthor": False
                        },
                        {
                            "provider_id": "fe03fcb5-2291-4a3e-a936-d173c4f4171a",
                            "provider_type": "workflow",
                            "provider_name": "addSpecialInstructions",
                            "tool_name": "addSpecialInstructions",
                            "tool_label": "addSpecialInstructions",
                            "tool_parameters": {
                                "specialInstructions": ""
                            },
                            "enabled": True,
                            "isDeleted": False,
                            "notAuthor": False
                        },
                        {
                            "provider_id": "841dfde5-089c-491d-8560-daeb2ece778d",
                            "provider_type": "workflow",
                            "provider_name": "selectWingType",
                            "tool_name": "selectWingType",
                            "tool_label": "selectWingType",
                            "tool_parameters": {
                                "wingType": ""
                            },
                            "enabled": True,
                            "isDeleted": False,
                            "notAuthor": False
                        },
                        {
                            "provider_id": "a4f2ba53-568b-481b-aea2-022aaf0b36e9",
                            "provider_type": "workflow",
                            "provider_name": "selectWingSauce",
                            "tool_name": "selectWingSauce",
                            "tool_label": "selectWingSauce",
                            "tool_parameters": {
                                "wingSauce": "",
                                "quantityOfSauce": "",
                                "wingType": ""
                            },
                            "enabled": True,
                            "isDeleted": False,
                            "notAuthor": False
                        },
                        {
                            "provider_id": "25eea377-d8de-4f85-8dbd-52f25d540cdf",
                            "provider_type": "workflow",
                            "provider_name": "selectSodas",
                            "tool_name": "selectSodas",
                            "tool_label": "selectSodas",
                            "tool_parameters": {
                                "firstDrink": "",
                                "secondDrink": "",
                                "thirdDrink": "",
                                "fourthDrink": ""
                            },
                            "enabled": True,
                            "isDeleted": False,
                            "notAuthor": False
                        },
                        {
                            "provider_id": "c9fd2140-9405-4da4-990e-8e165a479de3",
                            "provider_type": "workflow",
                            "provider_name": "finalOrderAssembly",
                            "tool_name": "finalOrderAssembly",
                            "tool_label": "finalOrderAssembly",
                            "tool_parameters": {
                                "pizzaSize": "",
                                "typeOfCrust": "",
                                "baseSauce": "",
                                "baseCheese": "",
                                "userWantsToppings": "",
                                "pizzaToppings": "",
                                "pizzaToppingsQuantity": "",
                                "pizzaToppingsDirection": "",
                                "userWantsExtraToppings": "",
                                "extraPizzaToppings": "",
                                "extraPizzaToppingsQuantity": "",
                                "extraPizzaToppingsDirection": "",
                                "pizzaSpecialInstructions": "",
                                "wingType": "",
                                "wingSauce": "",
                                "wingSauceQuantity": "",
                                "firstDrink": "",
                                "secondDrink": "",
                                "thirdDrink": "",
                                "fourthDrink": "",
                                "storeId": ""
                            },
                            "enabled": True,
                            "isDeleted": False,
                            "notAuthor": False
                        },
                        {
                            "provider_id": "a5258f3b-844f-4c9f-b5d6-e27336cb4cf6",
                            "provider_type": "workflow",
                            "provider_name": "editCart",
                            "tool_name": "editCart",
                            "tool_label": "editCart",
                            "tool_parameters": {
                                "pizzaSize": "",
                                "typeOfCrust": "",
                                "baseSauce": "",
                                "baseCheese": "",
                                "userWantsToppings": "",
                                "pizzaToppings": "",
                                "pizzaToppingsQuantity": "",
                                "pizzaToppingsDirection": "",
                                "userWantsExtraToppings": "",
                                "extraPizzaToppings": "",
                                "extraPizzaToppingsQuantity": "",
                                "extraPizzaToppingsDirection": "",
                                "pizzaSpecialInstructions": "",
                                "wingType": "",
                                "wingSauce": "",
                                "wingSauceQuantity": "",
                                "firstDrink": "",
                                "secondDrink": "",
                                "thirdDrink": "",
                                "fourthDrink": "",
                                "storeId": ""
                            },
                            "enabled": True,
                            "isDeleted": False,
                            "notAuthor": False
                        },
                        {
                            "provider_id": "e483bbe4-641a-4b0f-afdd-be81ec2a64f4",
                            "provider_type": "workflow",
                            "provider_name": "POST Future Store Hours",
                            "tool_name": "postFutureStoreHours",
                            "tool_label": "POST Future Store Hours",
                            "tool_parameters": {
                                "storeId": "",
                                "orderType": ""
                            },
                            "enabled": True,
                            "isDeleted": False,
                            "notAuthor": False
                        },
                        {
                            "provider_id": "236a0cf8-b76d-427b-a788-fb1148bba359",
                            "provider_type": "workflow",
                            "provider_name": "completeOrder",
                            "tool_name": "completeOrder",
                            "tool_label": "completeOrder",
                            "tool_parameters": {
                                "storeId": "",
                                "customerFirstName": "",
                                "customerLastName": "",
                                "customerEmail": "",
                                "customerPhone": ""
                            },
                            "enabled": True,
                            "isDeleted": False,
                            "notAuthor": False
                        },
                        {
                            "provider_id": "94b29ce1-ed91-42c5-9df9-971398998566",
                            "provider_type": "workflow",
                            "provider_name": "greetUser",
                            "tool_name": "greetUser",
                            "tool_label": "greetUser",
                            "tool_parameters": {},
                            "enabled": False,
                            "isDeleted": False,
                            "notAuthor": False
                        }
                    ],
                    "prompt": None
                },
                "dataset_configs": {
                    "retrieval_model": "single",
                    "datasets": {
                        "datasets": [
                            {
                                "dataset": {
                                    "enabled": True,
                                    "id": "fe579335-e455-4501-939e-585319f23fc8"
                                }
                            }
                        ]
                    }
                },
                "file_upload": {
                    "image": {
                        "enabled": False,
                        "number_limits": 3,
                        "detail": "high",
                        "transfer_methods": [
                            "remote_url",
                            "local_file"
                        ]
                    }
                },
                "annotation_reply": {
                    "enabled": False
                },
                "supportAnnotation": True,
                "appId": "cdf4e5b4-7411-4362-a073-1af8046a8c9b",
                "supportCitationHitInfo": True,
                "opening_workflow": "",
                "model": {
                    "provider": "openai",
                    "name": "gpt-4o",
                    "mode": "chat",
                    "completion_params": {
                        "stop": []
                    }
                }
            }
        }

    response = requests.post(url, headers=headers, json=data)
    response_data = []
    final_message = ""
    token_details = {}
    for line in response.iter_lines():
        if line:
            chuck_response = line.decode().strip().removeprefix("data: ")
            chunk_response_json = json.loads(chuck_response)
            event = chunk_response_json.get("event")
            if event == "agent_message":
                response_data.append(chunk_response_json["answer"])
            elif event == "message_end":
                final_message = "".join(response_data)
                token_details = chunk_response_json

    # print("************ Final response:", final_message)
    # print("************ Token details:", token_details)
    return final_message, token_details


if __name__ == "__main__":
    conversation_id = "8fb1fa3d-c89c-4039-9d05-c880232fa61e"
    token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IjFNS1BrVzhqbDl1Mk8zZlE1MjdNNSJ9.eyJ1c2VyX3JvbGUiOlsiU0FNTFVzZXJSb2xlIl0sIm1lbWJlcl9vZl9ncm91cHMiOiJfYXBwX2F1dGgwX3Rlc3QiLCJ1c2VyX25hbWUiOiJrcmlzaG5hLnNpbmdoQHF1YW50LmFpIiwidXNlcl9tYWlsIjoia3Jpc2huYS5zaW5naEBxdWFudC5haSIsImlzcyI6Imh0dHBzOi8vZGV2LW90MWpjdnZ3emxsejR4dXUudXMuYXV0aDAuY29tLyIsInN1YiI6InNhbWxwfFNBTUwtSnVtcGNsb3VkfGtyaXNobmEuc2luZ2hAcXVhbnQuYWkiLCJhdWQiOlsiaHR0cHM6Ly9hcHAuZW5nLnF1YW50LmFpLyIsImh0dHBzOi8vZGV2LW90MWpjdnZ3emxsejR4dXUudXMuYXV0aDAuY29tL3VzZXJpbmZvIl0sImlhdCI6MTc0NjYxNTA4MywiZXhwIjoxNzQ2NjIyMjgzLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIiwiYXpwIjoiY3d1VEZHQW83SmhjVlRQeVpDbEF6S1V5MlBIcWd6ZlEiLCJwZXJtaXNzaW9ucyI6WyJyZWFkOndlYiJdfQ.jMl3QHcfNlKew_pALn-NgJglHalOqGsh19WBivDapHjzJMmrs2lcGx2omnyj6RsreWhuXuc5qsLgWayhUrWyl9o2ZId9HUNLK9vEfKUXdYCW5qTop4HoPJll-2YehIWK_Nco9YpFwmZia2x4y_7dEFYrialcQMEnJgV7OZKoQmZFgA8Qlm3l8_oxtMz2qAtcbuuJpBs8nhL2TN6tgrG3MmIDqGh6YDGF9AJDcSNDFaL92k-irMjmS_VMRIaxNkqjMhceyzKAJ1wOIik-8Yp8EhsI1LL1DUgDCzMjJIoUCYgDmfk5iZQqM6I3qrqTW7RUKIYMUXN5p4_3oQYe-SV3ng"
    final_message, token_details = query_chatbot(conversation_id=conversation_id, bearer_token=token, query="Wow so many pizza?")
    print(f"final_message: {final_message}")
    print(f"token_details: {token_details}")


