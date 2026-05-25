/*
  Double Accelerated Math Practice

  Static GitHub Pages app. No server and no external libraries required.

  Add a new test by adding one object to TESTS. Each question has:
  prompt, preferredAnswer, acceptedAnswers, optional topic, and optional visual.

  This file intentionally avoids newer JavaScript syntax so older local
  node --check versions can parse it.
*/

var APP_VERSION = "2026-05-25-v5";
var RESULT_COOKIE = "mathPracticeResultsV5";
var STORAGE_KEY = "mathPracticeStateV5";

var TESTS = [
  {
    "id": "double-accelerated-practice-a",
    "title": "Practice Test A",
    "description": "30-question readiness practice, adapted to the online answer-only format.",
    "questions": [
      {
        "prompt": "6 × [48 ÷ (7 + 5) + 9] - 18",
        "preferredAnswer": "60",
        "acceptedAnswers": [
          "60"
        ],
        "topic": "Order of operations; whole-number operations",
        "looseAcceptedAnswers": [
          "60"
        ]
      },
      {
        "prompt": "(4 × 2)(3 × 2)",
        "preferredAnswer": "48",
        "acceptedAnswers": [
          "48"
        ],
        "topic": "Order of operations",
        "looseAcceptedAnswers": [
          "48"
        ]
      },
      {
        "prompt": "A school prints 327 packets. Each packet has 46 pages. How many pages are printed in all?",
        "preferredAnswer": "15,042 pages",
        "acceptedAnswers": [
          "15,042 pages"
        ],
        "topic": "Multi-digit multiplication",
        "looseAcceptedAnswers": [
          "15,042 pages",
          "15042 pages"
        ]
      },
      {
        "prompt": "4,375 ÷ 25",
        "preferredAnswer": "175",
        "acceptedAnswers": [
          "175"
        ],
        "topic": "Multi-digit division",
        "looseAcceptedAnswers": [
          "175"
        ]
      },
      {
        "prompt": "0.6 ÷ 0.001",
        "preferredAnswer": "600",
        "acceptedAnswers": [
          "600"
        ],
        "topic": "Powers of ten; decimal division",
        "looseAcceptedAnswers": [
          "600"
        ]
      },
      {
        "prompt": "In the number 62.486, the digit 4 has a value of 0.4 and the digit 8 has a value of 0.08. How many times as great is 0.4 as 0.08?",
        "preferredAnswer": "5",
        "acceptedAnswers": [
          "5"
        ],
        "topic": "Decimal place value comparison",
        "looseAcceptedAnswers": [
          "5"
        ]
      },
      {
        "prompt": "18.75 - 6.8 + 2.35",
        "preferredAnswer": "14.3",
        "acceptedAnswers": [
          "14.3"
        ],
        "topic": "Decimal addition and subtraction",
        "looseAcceptedAnswers": [
          "14.3",
          "14.30"
        ]
      },
      {
        "prompt": "A 12.6-meter ribbon is cut into pieces that are each 0.7 meter long. How many pieces can be cut?",
        "preferredAnswer": "18 pieces",
        "acceptedAnswers": [
          "18 pieces"
        ],
        "topic": "Decimal division",
        "looseAcceptedAnswers": [
          "18 pieces",
          "18 piece"
        ]
      },
      {
        "prompt": "Mia buys 3 notebooks for $4.85 each and 2 pens for $1.75 each. She pays with a $20 bill. How much change should she get?",
        "preferredAnswer": "$1.95",
        "acceptedAnswers": [
          "$1.95"
        ],
        "topic": "Multi-step decimal money problem",
        "looseAcceptedAnswers": [
          "$1.95",
          "1.95 dollars",
          "1.95"
        ]
      },
      {
        "prompt": "What is the difference between 5/8 and 7/12?",
        "preferredAnswer": "1/24",
        "acceptedAnswers": [
          "1/24"
        ],
        "topic": "Compare/subtract fractions with unlike denominators",
        "looseAcceptedAnswers": [
          "1/24"
        ]
      },
      {
        "prompt": "3/4 + 5/6 - 1/3",
        "preferredAnswer": "1 1/4",
        "acceptedAnswers": [
          "1 1/4"
        ],
        "topic": "Add/subtract fractions with unlike denominators",
        "looseAcceptedAnswers": [
          "1 1/4",
          "5/4",
          "1.25",
          "1 and 1/4"
        ]
      },
      {
        "prompt": "4 2/3 - 1 5/6",
        "preferredAnswer": "2 5/6",
        "acceptedAnswers": [
          "2 5/6"
        ],
        "topic": "Mixed-number subtraction",
        "looseAcceptedAnswers": [
          "2 5/6",
          "17/6",
          "2 and 5/6"
        ]
      },
      {
        "prompt": "One batch of trail mix uses 1 1/2 cups of oats and 3/4 cup of raisins. How many total cups of ingredients are needed for 2 1/2 batches?",
        "preferredAnswer": "5 5/8 cups",
        "acceptedAnswers": [
          "5 5/8 cups"
        ],
        "topic": "Multiplication of mixed numbers; scaling",
        "looseAcceptedAnswers": [
          "5 5/8 cups",
          "45/8 cups",
          "5.625 cups",
          "5 and 5/8 cups"
        ]
      },
      {
        "prompt": "2.4 × 35",
        "preferredAnswer": "84",
        "acceptedAnswers": [
          "84"
        ],
        "topic": "Whole-number-times-decimal multiplication",
        "looseAcceptedAnswers": [
          "84"
        ]
      },
      {
        "prompt": "A rectangle is 3/4 meter long and 2/3 meter wide. What is its area?",
        "preferredAnswer": "1/2 square meter",
        "acceptedAnswers": [
          "1/2 square meter"
        ],
        "topic": "Area with fraction side lengths",
        "looseAcceptedAnswers": [
          "1/2 square meter",
          "0.5 square meter",
          "1/2 square meters",
          "0.5 square meters",
          "1/2 m^2",
          "0.5 m^2",
          "1/2 m²",
          "0.5 m²"
        ]
      },
      {
        "prompt": "You have 4 liters of juice. Each serving is 1/3 liter. How many servings can you pour?",
        "preferredAnswer": "12 servings",
        "acceptedAnswers": [
          "12 servings"
        ],
        "topic": "Divide whole number by unit fraction",
        "looseAcceptedAnswers": [
          "12 servings",
          "12 serving"
        ]
      },
      {
        "prompt": "A 1/2 cup of sugar is split equally among 4 small recipes. How much sugar does each recipe get?",
        "preferredAnswer": "1/8 cup",
        "acceptedAnswers": [
          "1/8 cup"
        ],
        "topic": "Divide unit fraction by whole number",
        "looseAcceptedAnswers": [
          "1/8 cup",
          "0.125 cup"
        ]
      },
      {
        "prompt": "Seven granola bars are shared equally by 4 children. How many granola bars does each child get?",
        "preferredAnswer": "1 3/4 granola bars",
        "acceptedAnswers": [
          "1 3/4 granola bars"
        ],
        "topic": "Division as fraction",
        "looseAcceptedAnswers": [
          "1 3/4 granola bars",
          "7/4 granola bars",
          "1.75 granola bars",
          "1 and 3/4 granola bars"
        ]
      },
      {
        "prompt": "How much greater is 3/4 than 0.705?",
        "preferredAnswer": "0.045",
        "acceptedAnswers": [
          "0.045"
        ],
        "topic": "Fraction/decimal conversion and subtraction",
        "looseAcceptedAnswers": [
          "0.045",
          "45/1000",
          "9/200"
        ]
      },
      {
        "prompt": "Use 1 foot = 12 inches and 1 yard = 3 feet. Add 5 feet 8 inches + 2 yards 1 foot. Give your answer in inches.",
        "preferredAnswer": "152 inches",
        "acceptedAnswers": [
          "152 inches"
        ],
        "topic": "Measurement conversion",
        "looseAcceptedAnswers": [
          "152 inches",
          "152 in",
          "152 in."
        ]
      },
      {
        "prompt": "A movie starts at 1:45 PM. Previews last 18 minutes, and the movie lasts 1 hour 52 minutes. How many minutes after 1:45 PM does everything end?",
        "preferredAnswer": "130 minutes",
        "acceptedAnswers": [
          "130 minutes"
        ],
        "topic": "Elapsed time",
        "looseAcceptedAnswers": [
          "130 minutes",
          "130 minute",
          "130 min",
          "130"
        ]
      },
      {
        "prompt": "A rectangular garden is 28 inches long and 20 inches wide. Wes buys 9 feet of fence. If he fences the garden once around, how many inches of fence are left?",
        "preferredAnswer": "12 inches",
        "acceptedAnswers": [
          "12 inches"
        ],
        "topic": "Perimeter with unit conversion",
        "looseAcceptedAnswers": [
          "12 inches",
          "12 in",
          "12 in."
        ]
      },
      {
        "prompt": "Three adjacent angles form a straight line. Two of the angles measure 43 degrees and 68 degrees. What is the measure of the third angle?",
        "preferredAnswer": "69 degrees",
        "acceptedAnswers": [
          "69 degrees"
        ],
        "topic": "Angle measure; straight angle",
        "looseAcceptedAnswers": [
          "69 degrees",
          "69°"
        ]
      },
      {
        "prompt": "A rectangle is 17 centimeters long and 9 centimeters wide. What is its area?",
        "preferredAnswer": "153 square centimeters",
        "acceptedAnswers": [
          "153 square centimeters"
        ],
        "topic": "Area of a rectangle",
        "looseAcceptedAnswers": [
          "153 square centimeters",
          "153 sq cm",
          "153 cm^2",
          "153 cm²"
        ]
      },
      {
        "prompt": "Points A(2,1), B(2,5), C(7,5), and D(7,1) are connected in order on a graph. What is the area of the shape?",
        "preferredAnswer": "20 square units",
        "acceptedAnswers": [
          "20 square units"
        ],
        "topic": "Coordinate plane; area of rectangle",
        "looseAcceptedAnswers": [
          "20 square units",
          "20 units^2",
          "20 unit^2",
          "20 units²"
        ]
      },
      {
        "prompt": "A rectangular prism has length 8 cm, width 5 cm, and height 3 cm. What is its volume?",
        "preferredAnswer": "120 cubic centimeters",
        "acceptedAnswers": [
          "120 cubic centimeters"
        ],
        "topic": "Volume of rectangular prism",
        "looseAcceptedAnswers": [
          "120 cubic centimeters",
          "120 cubic cm",
          "120 cm^3",
          "120 cm³"
        ]
      },
      {
        "prompt": "A solid is made from two rectangular prisms with no overlap. Prism 1 is 6 inches by 4 inches by 3 inches. Prism 2 is 2 inches by 4 inches by 5 inches. What is the total volume?",
        "preferredAnswer": "112 cubic inches",
        "acceptedAnswers": [
          "112 cubic inches"
        ],
        "topic": "Additive volume",
        "looseAcceptedAnswers": [
          "112 cubic inches",
          "112 in^3",
          "112 in³"
        ]
      },
      {
        "prompt": "Pattern A starts at 4 and adds 6 each time. What is the 10th term of Pattern A?",
        "preferredAnswer": "58",
        "acceptedAnswers": [
          "58"
        ],
        "topic": "Numerical patterns",
        "looseAcceptedAnswers": [
          "58"
        ]
      },
      {
        "prompt": "A field trip has 96 students and 8 adults. Each bus seats 36 people. Each bus costs $275. Admission costs $8.50 per student and $12 per adult. The class has already raised $1,000. How much more money is needed?",
        "preferredAnswer": "$737",
        "acceptedAnswers": [
          "$737"
        ],
        "topic": "Multi-step word problem; operations with decimals",
        "looseAcceptedAnswers": [
          "$737",
          "737 dollars",
          "737"
        ]
      },
      {
        "prompt": "Use the line plot below. It shows ribbon lengths in inches. What is the total length of all the ribbons in inches?",
        "preferredAnswer": "6 1/4 inches",
        "acceptedAnswers": [
          "6 1/4 inches"
        ],
        "topic": "Measurement data",
        "visual": {
          "type": "linePlot",
          "title": "Line plot: ribbon lengths in inches",
          "columns": [
            {
              "label": "1/4",
              "marks": 2
            },
            {
              "label": "1/2",
              "marks": 3
            },
            {
              "label": "3/4",
              "marks": 3
            },
            {
              "label": "1",
              "marks": 2
            }
          ]
        },
        "looseAcceptedAnswers": [
          "6 1/4 inches",
          "25/4 inches",
          "6.25 inches",
          "6 and 1/4 inches",
          "6 1/4 in",
          "25/4 in",
          "6.25 in"
        ]
      }
    ]
  },
  {
    "id": "double-accelerated-practice-b",
    "title": "Practice Test B",
    "description": "Mixed operations, fraction fluency, measurement, and geometry practice.",
    "questions": [
      {
        "prompt": "8 × [72 ÷ (9 + 3) + 5] - 16",
        "preferredAnswer": "72",
        "acceptedAnswers": [
          "72"
        ],
        "topic": "Order of operations",
        "looseAcceptedAnswers": [
          "72"
        ]
      },
      {
        "prompt": "(5 × 3)(2 + 4)",
        "preferredAnswer": "90",
        "acceptedAnswers": [
          "90"
        ],
        "topic": "Order of operations",
        "looseAcceptedAnswers": [
          "90"
        ]
      },
      {
        "prompt": "A collector has 214 boxes of cards. Each box has 38 cards. How many cards are there?",
        "preferredAnswer": "8,132 cards",
        "acceptedAnswers": [
          "8,132 cards"
        ],
        "topic": "Multi-digit multiplication",
        "looseAcceptedAnswers": [
          "8,132 cards",
          "8132 cards"
        ]
      },
      {
        "prompt": "6,840 ÷ 45",
        "preferredAnswer": "152",
        "acceptedAnswers": [
          "152"
        ],
        "topic": "Multi-digit division",
        "looseAcceptedAnswers": [
          "152"
        ]
      },
      {
        "prompt": "7.2 ÷ 0.01",
        "preferredAnswer": "720",
        "acceptedAnswers": [
          "720"
        ],
        "topic": "Powers of ten; decimal division",
        "looseAcceptedAnswers": [
          "720"
        ]
      },
      {
        "prompt": "In the number 45.808, the first 8 has a value of 0.8 and the second 8 has a value of 0.008. How many times as great is 0.8 as 0.008?",
        "preferredAnswer": "100",
        "acceptedAnswers": [
          "100"
        ],
        "topic": "Decimal place value comparison",
        "looseAcceptedAnswers": [
          "100"
        ]
      },
      {
        "prompt": "26.4 - 7.85 + 3.6",
        "preferredAnswer": "22.15",
        "acceptedAnswers": [
          "22.15"
        ],
        "topic": "Decimal addition and subtraction",
        "looseAcceptedAnswers": [
          "22.15"
        ]
      },
      {
        "prompt": "A 15.75-meter ribbon is cut into pieces that are each 0.25 meter long. How many pieces can be cut?",
        "preferredAnswer": "63 pieces",
        "acceptedAnswers": [
          "63 pieces"
        ],
        "topic": "Decimal division",
        "looseAcceptedAnswers": [
          "63 pieces",
          "63 piece"
        ]
      },
      {
        "prompt": "Nora buys 4 journals for $3.75 each and 3 erasers for $1.20 each. She pays with a $20 bill. How much change should she get?",
        "preferredAnswer": "$1.40",
        "acceptedAnswers": [
          "$1.40"
        ],
        "topic": "Multi-step decimal money problem",
        "looseAcceptedAnswers": [
          "$1.40",
          "1.40 dollars",
          "1.4 dollars",
          "1.40",
          "1.4"
        ]
      },
      {
        "prompt": "What is the difference between 7/10 and 2/3?",
        "preferredAnswer": "1/30",
        "acceptedAnswers": [
          "1/30"
        ],
        "topic": "Compare/subtract fractions with unlike denominators",
        "looseAcceptedAnswers": [
          "1/30"
        ]
      },
      {
        "prompt": "2/3 + 3/4 - 1/6",
        "preferredAnswer": "1 1/4",
        "acceptedAnswers": [
          "1 1/4"
        ],
        "topic": "Add/subtract fractions with unlike denominators",
        "looseAcceptedAnswers": [
          "1 1/4",
          "5/4",
          "1.25",
          "1 and 1/4"
        ]
      },
      {
        "prompt": "6 1/4 - 2 5/8",
        "preferredAnswer": "3 5/8",
        "acceptedAnswers": [
          "3 5/8"
        ],
        "topic": "Mixed-number subtraction",
        "looseAcceptedAnswers": [
          "3 5/8",
          "29/8",
          "3.625",
          "3 and 5/8"
        ]
      },
      {
        "prompt": "One batch of snack mix uses 2/3 cup of cereal and 1/2 cup of nuts. How many total cups of ingredients are needed for 3 batches?",
        "preferredAnswer": "3 1/2 cups",
        "acceptedAnswers": [
          "3 1/2 cups"
        ],
        "topic": "Fraction multiplication; scaling",
        "looseAcceptedAnswers": [
          "3 1/2 cups",
          "7/2 cups",
          "3.5 cups",
          "3 and 1/2 cups"
        ]
      },
      {
        "prompt": "1.8 × 45",
        "preferredAnswer": "81",
        "acceptedAnswers": [
          "81"
        ],
        "topic": "Whole-number-times-decimal multiplication",
        "looseAcceptedAnswers": [
          "81"
        ]
      },
      {
        "prompt": "A rectangle is 5/6 meter long and 3/5 meter wide. What is its area?",
        "preferredAnswer": "1/2 square meter",
        "acceptedAnswers": [
          "1/2 square meter"
        ],
        "topic": "Area with fraction side lengths",
        "looseAcceptedAnswers": [
          "1/2 square meter",
          "0.5 square meter",
          "1/2 m^2",
          "0.5 m^2",
          "1/2 m²",
          "0.5 m²"
        ]
      },
      {
        "prompt": "You have 3 liters of juice. Each serving is 1/4 liter. How many servings can you pour?",
        "preferredAnswer": "12 servings",
        "acceptedAnswers": [
          "12 servings"
        ],
        "topic": "Divide whole number by unit fraction",
        "looseAcceptedAnswers": [
          "12 servings",
          "12 serving"
        ]
      },
      {
        "prompt": "A 3/4 cup of sugar is split equally among 6 small recipes. How much sugar does each recipe get?",
        "preferredAnswer": "1/8 cup",
        "acceptedAnswers": [
          "1/8 cup"
        ],
        "topic": "Divide fraction by whole number",
        "looseAcceptedAnswers": [
          "1/8 cup",
          "0.125 cup"
        ]
      },
      {
        "prompt": "Nine granola bars are shared equally by 8 children. How many granola bars does each child get?",
        "preferredAnswer": "1 1/8 granola bars",
        "acceptedAnswers": [
          "1 1/8 granola bars"
        ],
        "topic": "Division as fraction",
        "looseAcceptedAnswers": [
          "1 1/8 granola bars",
          "9/8 granola bars",
          "1.125 granola bars",
          "1 and 1/8 granola bars"
        ]
      },
      {
        "prompt": "How much greater is 0.83 than 4/5?",
        "preferredAnswer": "0.03",
        "acceptedAnswers": [
          "0.03"
        ],
        "topic": "Fraction/decimal conversion and subtraction",
        "looseAcceptedAnswers": [
          "0.03",
          "3/100"
        ]
      },
      {
        "prompt": "Use 1 foot = 12 inches and 1 yard = 3 feet. Add 3 yards 2 feet + 4 feet 6 inches. Give your answer in inches.",
        "preferredAnswer": "186 inches",
        "acceptedAnswers": [
          "186 inches"
        ],
        "topic": "Measurement conversion",
        "looseAcceptedAnswers": [
          "186 inches",
          "186 in",
          "186 in."
        ]
      },
      {
        "prompt": "A workshop starts at 10:20 AM. The first activity lasts 35 minutes, and the second lasts 1 hour 15 minutes. How many minutes after 10:20 AM does everything end?",
        "preferredAnswer": "110 minutes",
        "acceptedAnswers": [
          "110 minutes"
        ],
        "topic": "Elapsed time",
        "looseAcceptedAnswers": [
          "110 minutes",
          "110 minute",
          "110 min",
          "110"
        ]
      },
      {
        "prompt": "A rectangular garden is 35 inches long and 23 inches wide. Wes buys 10 feet of fence. If he fences the garden once around, how many inches of fence are left?",
        "preferredAnswer": "4 inches",
        "acceptedAnswers": [
          "4 inches"
        ],
        "topic": "Perimeter with unit conversion",
        "looseAcceptedAnswers": [
          "4 inches",
          "4 in",
          "4 in."
        ]
      },
      {
        "prompt": "Three adjacent angles form a straight line. Two of the angles measure 52 degrees and 77 degrees. What is the measure of the third angle?",
        "preferredAnswer": "51 degrees",
        "acceptedAnswers": [
          "51 degrees"
        ],
        "topic": "Angle measure; straight angle",
        "looseAcceptedAnswers": [
          "51 degrees",
          "51°"
        ]
      },
      {
        "prompt": "A rectangle is 24 centimeters long and 13 centimeters wide. What is its area?",
        "preferredAnswer": "312 square centimeters",
        "acceptedAnswers": [
          "312 square centimeters"
        ],
        "topic": "Area of a rectangle",
        "looseAcceptedAnswers": [
          "312 square centimeters",
          "312 sq cm",
          "312 cm^2",
          "312 cm²"
        ]
      },
      {
        "prompt": "Points A(1,2), B(1,8), C(6,8), and D(6,2) are connected in order on a graph. What is the area of the shape?",
        "preferredAnswer": "30 square units",
        "acceptedAnswers": [
          "30 square units"
        ],
        "topic": "Coordinate plane; area of rectangle",
        "looseAcceptedAnswers": [
          "30 square units",
          "30 units^2",
          "30 units²"
        ]
      },
      {
        "prompt": "A rectangular prism has length 7 cm, width 4 cm, and height 6 cm. What is its volume?",
        "preferredAnswer": "168 cubic centimeters",
        "acceptedAnswers": [
          "168 cubic centimeters"
        ],
        "topic": "Volume of rectangular prism",
        "looseAcceptedAnswers": [
          "168 cubic centimeters",
          "168 cubic cm",
          "168 cm^3",
          "168 cm³"
        ]
      },
      {
        "prompt": "A solid is made from two rectangular prisms with no overlap. Prism 1 is 5 inches by 3 inches by 4 inches. Prism 2 is 2 inches by 3 inches by 7 inches. What is the total volume?",
        "preferredAnswer": "102 cubic inches",
        "acceptedAnswers": [
          "102 cubic inches"
        ],
        "topic": "Additive volume",
        "looseAcceptedAnswers": [
          "102 cubic inches",
          "102 in^3",
          "102 in³"
        ]
      },
      {
        "prompt": "Pattern A starts at 9 and adds 4 each time. What is the 12th term of Pattern A?",
        "preferredAnswer": "53",
        "acceptedAnswers": [
          "53"
        ],
        "topic": "Numerical patterns",
        "looseAcceptedAnswers": [
          "53"
        ]
      },
      {
        "prompt": "A field trip has 84 students and 6 adults. Each bus seats 32 people. Each bus costs $240. Admission costs $7.50 per student and $11 per adult. The class has already raised $900. How much more money is needed?",
        "preferredAnswer": "$516",
        "acceptedAnswers": [
          "$516"
        ],
        "topic": "Multi-step word problem; operations with decimals",
        "looseAcceptedAnswers": [
          "$516",
          "516 dollars",
          "516"
        ]
      },
      {
        "prompt": "Use the line plot below. It shows ribbon lengths in inches. What is the total length of all the ribbons in inches?",
        "preferredAnswer": "5 3/4 inches",
        "acceptedAnswers": [
          "5 3/4 inches"
        ],
        "topic": "Measurement data",
        "visual": {
          "type": "linePlot",
          "title": "Line plot: ribbon lengths in inches",
          "columns": [
            {
              "label": "1/4",
              "marks": 3
            },
            {
              "label": "1/2",
              "marks": 2
            },
            {
              "label": "3/4",
              "marks": 4
            },
            {
              "label": "1",
              "marks": 1
            }
          ]
        },
        "looseAcceptedAnswers": [
          "5 3/4 inches",
          "23/4 inches",
          "5.75 inches",
          "5 and 3/4 inches",
          "5 3/4 in",
          "23/4 in",
          "5.75 in"
        ]
      }
    ]
  },
  {
    "id": "double-accelerated-practice-c",
    "title": "Practice Test C",
    "description": "Decimals, geometry, coordinate-plane area, volume, and fraction operations.",
    "questions": [
      {
        "prompt": "9 × [64 ÷ (5 + 3) + 7] - 20",
        "preferredAnswer": "115",
        "acceptedAnswers": [
          "115"
        ],
        "topic": "Order of operations",
        "looseAcceptedAnswers": [
          "115"
        ]
      },
      {
        "prompt": "(7 - 2)(6 × 3)",
        "preferredAnswer": "90",
        "acceptedAnswers": [
          "90"
        ],
        "topic": "Order of operations",
        "looseAcceptedAnswers": [
          "90"
        ]
      },
      {
        "prompt": "A school prints 506 packets. Each packet has 29 pages. How many pages are printed in all?",
        "preferredAnswer": "14,674 pages",
        "acceptedAnswers": [
          "14,674 pages"
        ],
        "topic": "Multi-digit multiplication",
        "looseAcceptedAnswers": [
          "14,674 pages",
          "14674 pages"
        ]
      },
      {
        "prompt": "8,925 ÷ 35",
        "preferredAnswer": "255",
        "acceptedAnswers": [
          "255"
        ],
        "topic": "Multi-digit division",
        "looseAcceptedAnswers": [
          "255"
        ]
      },
      {
        "prompt": "3.4 ÷ 0.001",
        "preferredAnswer": "3,400",
        "acceptedAnswers": [
          "3,400"
        ],
        "topic": "Powers of ten; decimal division",
        "looseAcceptedAnswers": [
          "3,400",
          "3400"
        ]
      },
      {
        "prompt": "In the number 72.606, the first 6 has a value of 0.6 and the second 6 has a value of 0.006. How many times as great is 0.6 as 0.006?",
        "preferredAnswer": "100",
        "acceptedAnswers": [
          "100"
        ],
        "topic": "Decimal place value comparison",
        "looseAcceptedAnswers": [
          "100"
        ]
      },
      {
        "prompt": "50 - 12.75 + 0.8",
        "preferredAnswer": "38.05",
        "acceptedAnswers": [
          "38.05"
        ],
        "topic": "Decimal addition and subtraction",
        "looseAcceptedAnswers": [
          "38.05"
        ]
      },
      {
        "prompt": "A 9.6-meter ribbon is cut into pieces that are each 0.8 meter long. How many pieces can be cut?",
        "preferredAnswer": "12 pieces",
        "acceptedAnswers": [
          "12 pieces"
        ],
        "topic": "Decimal division",
        "looseAcceptedAnswers": [
          "12 pieces",
          "12 piece"
        ]
      },
      {
        "prompt": "Lena buys 5 folders for $2.65 each and 4 markers for $1.10 each. She pays with a $25 bill. How much change should she get?",
        "preferredAnswer": "$7.35",
        "acceptedAnswers": [
          "$7.35"
        ],
        "topic": "Multi-step decimal money problem",
        "looseAcceptedAnswers": [
          "$7.35",
          "7.35 dollars",
          "7.35"
        ]
      },
      {
        "prompt": "What is the difference between 3/5 and 7/12?",
        "preferredAnswer": "1/60",
        "acceptedAnswers": [
          "1/60"
        ],
        "topic": "Compare/subtract fractions with unlike denominators",
        "looseAcceptedAnswers": [
          "1/60"
        ]
      },
      {
        "prompt": "5/8 + 2/3 - 1/4",
        "preferredAnswer": "1 1/24",
        "acceptedAnswers": [
          "1 1/24"
        ],
        "topic": "Add/subtract fractions with unlike denominators",
        "looseAcceptedAnswers": [
          "1 1/24",
          "25/24",
          "1 and 1/24"
        ]
      },
      {
        "prompt": "7 1/2 - 3 3/4",
        "preferredAnswer": "3 3/4",
        "acceptedAnswers": [
          "3 3/4"
        ],
        "topic": "Mixed-number subtraction",
        "looseAcceptedAnswers": [
          "3 3/4",
          "15/4",
          "3.75",
          "3 and 3/4"
        ]
      },
      {
        "prompt": "One batch of trail mix uses 3/4 cup of oats and 2/5 cup of raisins. How many total cups of ingredients are needed for 5 batches?",
        "preferredAnswer": "5 3/4 cups",
        "acceptedAnswers": [
          "5 3/4 cups"
        ],
        "topic": "Fraction multiplication; scaling",
        "looseAcceptedAnswers": [
          "5 3/4 cups",
          "23/4 cups",
          "5.75 cups",
          "5 and 3/4 cups"
        ]
      },
      {
        "prompt": "3.6 × 25",
        "preferredAnswer": "90",
        "acceptedAnswers": [
          "90"
        ],
        "topic": "Whole-number-times-decimal multiplication",
        "looseAcceptedAnswers": [
          "90"
        ]
      },
      {
        "prompt": "A rectangle is 4/5 meter long and 5/8 meter wide. What is its area?",
        "preferredAnswer": "1/2 square meter",
        "acceptedAnswers": [
          "1/2 square meter"
        ],
        "topic": "Area with fraction side lengths",
        "looseAcceptedAnswers": [
          "1/2 square meter",
          "0.5 square meter",
          "1/2 m^2",
          "0.5 m^2",
          "1/2 m²",
          "0.5 m²"
        ]
      },
      {
        "prompt": "You have 5 liters of juice. Each serving is 1/5 liter. How many servings can you pour?",
        "preferredAnswer": "25 servings",
        "acceptedAnswers": [
          "25 servings"
        ],
        "topic": "Divide whole number by unit fraction",
        "looseAcceptedAnswers": [
          "25 servings",
          "25 serving"
        ]
      },
      {
        "prompt": "A 2/3 cup of sugar is split equally among 8 small recipes. How much sugar does each recipe get?",
        "preferredAnswer": "1/12 cup",
        "acceptedAnswers": [
          "1/12 cup"
        ],
        "topic": "Divide fraction by whole number",
        "looseAcceptedAnswers": [
          "1/12 cup"
        ]
      },
      {
        "prompt": "Eleven granola bars are shared equally by 4 children. How many granola bars does each child get?",
        "preferredAnswer": "2 3/4 granola bars",
        "acceptedAnswers": [
          "2 3/4 granola bars"
        ],
        "topic": "Division as fraction",
        "looseAcceptedAnswers": [
          "2 3/4 granola bars",
          "11/4 granola bars",
          "2.75 granola bars",
          "2 and 3/4 granola bars"
        ]
      },
      {
        "prompt": "How much greater is 7/8 than 0.82?",
        "preferredAnswer": "0.055",
        "acceptedAnswers": [
          "0.055"
        ],
        "topic": "Fraction/decimal conversion and subtraction",
        "looseAcceptedAnswers": [
          "0.055",
          "55/1000",
          "11/200"
        ]
      },
      {
        "prompt": "Use 1 foot = 12 inches and 1 yard = 3 feet. Add 4 yards 1 foot + 3 feet 9 inches. Give your answer in inches.",
        "preferredAnswer": "201 inches",
        "acceptedAnswers": [
          "201 inches"
        ],
        "topic": "Measurement conversion",
        "looseAcceptedAnswers": [
          "201 inches",
          "201 in",
          "201 in."
        ]
      },
      {
        "prompt": "A program starts at 2:10 PM. The opening lasts 22 minutes, and the main event lasts 1 hour 43 minutes. How many minutes after 2:10 PM does everything end?",
        "preferredAnswer": "125 minutes",
        "acceptedAnswers": [
          "125 minutes"
        ],
        "topic": "Elapsed time",
        "looseAcceptedAnswers": [
          "125 minutes",
          "125 minute",
          "125 min",
          "125"
        ]
      },
      {
        "prompt": "A rectangular garden is 42 inches long and 18 inches wide. Wes buys 11 feet of fence. If he fences the garden once around, how many inches of fence are left?",
        "preferredAnswer": "12 inches",
        "acceptedAnswers": [
          "12 inches"
        ],
        "topic": "Perimeter with unit conversion",
        "looseAcceptedAnswers": [
          "12 inches",
          "12 in",
          "12 in."
        ]
      },
      {
        "prompt": "Three adjacent angles form a straight line. Two of the angles measure 35 degrees and 91 degrees. What is the measure of the third angle?",
        "preferredAnswer": "54 degrees",
        "acceptedAnswers": [
          "54 degrees"
        ],
        "topic": "Angle measure; straight angle",
        "looseAcceptedAnswers": [
          "54 degrees",
          "54°"
        ]
      },
      {
        "prompt": "A rectangle is 21 centimeters long and 16 centimeters wide. What is its area?",
        "preferredAnswer": "336 square centimeters",
        "acceptedAnswers": [
          "336 square centimeters"
        ],
        "topic": "Area of a rectangle",
        "looseAcceptedAnswers": [
          "336 square centimeters",
          "336 sq cm",
          "336 cm^2",
          "336 cm²"
        ]
      },
      {
        "prompt": "Points A(3,2), B(3,9), C(10,9), and D(10,2) are connected in order on a graph. What is the area of the shape?",
        "preferredAnswer": "49 square units",
        "acceptedAnswers": [
          "49 square units"
        ],
        "topic": "Coordinate plane; area of rectangle",
        "looseAcceptedAnswers": [
          "49 square units",
          "49 units^2",
          "49 units²"
        ]
      },
      {
        "prompt": "A rectangular prism has length 9 cm, width 4 cm, and height 5 cm. What is its volume?",
        "preferredAnswer": "180 cubic centimeters",
        "acceptedAnswers": [
          "180 cubic centimeters"
        ],
        "topic": "Volume of rectangular prism",
        "looseAcceptedAnswers": [
          "180 cubic centimeters",
          "180 cubic cm",
          "180 cm^3",
          "180 cm³"
        ]
      },
      {
        "prompt": "A solid is made from two rectangular prisms with no overlap. Prism 1 is 4 inches by 4 inches by 6 inches. Prism 2 is 3 inches by 4 inches by 5 inches. What is the total volume?",
        "preferredAnswer": "156 cubic inches",
        "acceptedAnswers": [
          "156 cubic inches"
        ],
        "topic": "Additive volume",
        "looseAcceptedAnswers": [
          "156 cubic inches",
          "156 in^3",
          "156 in³"
        ]
      },
      {
        "prompt": "Pattern A starts at 2 and adds 7 each time. What is the 9th term of Pattern A?",
        "preferredAnswer": "58",
        "acceptedAnswers": [
          "58"
        ],
        "topic": "Numerical patterns",
        "looseAcceptedAnswers": [
          "58"
        ]
      },
      {
        "prompt": "A field trip has 72 students and 9 adults. Each bus seats 30 people. Each bus costs $310. Admission costs $9.25 per student and $13 per adult. The class has already raised $1,200. How much more money is needed?",
        "preferredAnswer": "$513",
        "acceptedAnswers": [
          "$513"
        ],
        "topic": "Multi-step word problem; operations with decimals",
        "looseAcceptedAnswers": [
          "$513",
          "513 dollars",
          "513"
        ]
      },
      {
        "prompt": "Use the line plot below. It shows ribbon lengths in inches. What is the total length of all the ribbons in inches?",
        "preferredAnswer": "6 1/4 inches",
        "acceptedAnswers": [
          "6 1/4 inches"
        ],
        "topic": "Measurement data",
        "visual": {
          "type": "linePlot",
          "title": "Line plot: ribbon lengths in inches",
          "columns": [
            {
              "label": "1/4",
              "marks": 2
            },
            {
              "label": "1/2",
              "marks": 4
            },
            {
              "label": "3/4",
              "marks": 1
            },
            {
              "label": "1",
              "marks": 3
            }
          ]
        },
        "looseAcceptedAnswers": [
          "6 1/4 inches",
          "25/4 inches",
          "6.25 inches",
          "6 and 1/4 inches",
          "6 1/4 in",
          "25/4 in",
          "6.25 in"
        ]
      }
    ]
  },
  {
    "id": "double-accelerated-practice-d",
    "title": "Practice Test D",
    "description": "More mixed-number arithmetic, decimal scaling, area, perimeter, and volume practice.",
    "questions": [
      {
        "prompt": "7 × [90 ÷ (6 + 9) + 8] - 11",
        "preferredAnswer": "87",
        "acceptedAnswers": [
          "87"
        ],
        "topic": "Order of operations",
        "looseAcceptedAnswers": [
          "87"
        ]
      },
      {
        "prompt": "(18 ÷ 3)(4 + 5)",
        "preferredAnswer": "54",
        "acceptedAnswers": [
          "54"
        ],
        "topic": "Order of operations",
        "looseAcceptedAnswers": [
          "54"
        ]
      },
      {
        "prompt": "A store packs 418 boxes. Each box has 37 stickers. How many stickers are packed?",
        "preferredAnswer": "15,466 stickers",
        "acceptedAnswers": [
          "15,466 stickers"
        ],
        "topic": "Multi-digit multiplication",
        "looseAcceptedAnswers": [
          "15,466 stickers",
          "15466 stickers"
        ]
      },
      {
        "prompt": "7,812 ÷ 42",
        "preferredAnswer": "186",
        "acceptedAnswers": [
          "186"
        ],
        "topic": "Multi-digit division",
        "looseAcceptedAnswers": [
          "186"
        ]
      },
      {
        "prompt": "0.48 ÷ 0.0001",
        "preferredAnswer": "4,800",
        "acceptedAnswers": [
          "4,800"
        ],
        "topic": "Powers of ten; decimal division",
        "looseAcceptedAnswers": [
          "4,800",
          "4800"
        ]
      },
      {
        "prompt": "In the number 31.515, the first 5 has a value of 0.5 and the second 5 has a value of 0.005. How many times as great is 0.5 as 0.005?",
        "preferredAnswer": "100",
        "acceptedAnswers": [
          "100"
        ],
        "topic": "Decimal place value comparison",
        "looseAcceptedAnswers": [
          "100"
        ]
      },
      {
        "prompt": "100.2 - 45.75 + 6.05",
        "preferredAnswer": "60.5",
        "acceptedAnswers": [
          "60.5"
        ],
        "topic": "Decimal addition and subtraction",
        "looseAcceptedAnswers": [
          "60.5",
          "60.50"
        ]
      },
      {
        "prompt": "An 18.9-meter ribbon is cut into pieces that are each 0.3 meter long. How many pieces can be cut?",
        "preferredAnswer": "63 pieces",
        "acceptedAnswers": [
          "63 pieces"
        ],
        "topic": "Decimal division",
        "looseAcceptedAnswers": [
          "63 pieces",
          "63 piece"
        ]
      },
      {
        "prompt": "Sam buys 6 notebooks for $2.95 each and 2 rulers for $1.15 each. He pays with a $30 bill. How much change should he get?",
        "preferredAnswer": "$10.00",
        "acceptedAnswers": [
          "$10.00"
        ],
        "topic": "Multi-step decimal money problem",
        "looseAcceptedAnswers": [
          "$10.00",
          "10 dollars",
          "10.00 dollars",
          "10",
          "10.00"
        ]
      },
      {
        "prompt": "What is the difference between 9/10 and 5/6?",
        "preferredAnswer": "1/15",
        "acceptedAnswers": [
          "1/15"
        ],
        "topic": "Compare/subtract fractions with unlike denominators",
        "looseAcceptedAnswers": [
          "1/15"
        ]
      },
      {
        "prompt": "7/10 + 3/5 - 1/4",
        "preferredAnswer": "1 1/20",
        "acceptedAnswers": [
          "1 1/20"
        ],
        "topic": "Add/subtract fractions with unlike denominators",
        "looseAcceptedAnswers": [
          "1 1/20",
          "21/20",
          "1.05",
          "1 and 1/20"
        ]
      },
      {
        "prompt": "8 3/4 - 5 5/6",
        "preferredAnswer": "2 11/12",
        "acceptedAnswers": [
          "2 11/12"
        ],
        "topic": "Mixed-number subtraction",
        "looseAcceptedAnswers": [
          "2 11/12",
          "35/12",
          "2 and 11/12"
        ]
      },
      {
        "prompt": "One batch of trail mix uses 1 1/4 cups of oats and 2/3 cup of raisins. How many total cups of ingredients are needed for 3 batches?",
        "preferredAnswer": "5 3/4 cups",
        "acceptedAnswers": [
          "5 3/4 cups"
        ],
        "topic": "Fraction multiplication; scaling",
        "looseAcceptedAnswers": [
          "5 3/4 cups",
          "23/4 cups",
          "5.75 cups",
          "5 and 3/4 cups"
        ]
      },
      {
        "prompt": "4.2 × 18",
        "preferredAnswer": "75.6",
        "acceptedAnswers": [
          "75.6"
        ],
        "topic": "Whole-number-times-decimal multiplication",
        "looseAcceptedAnswers": [
          "75.6"
        ]
      },
      {
        "prompt": "A rectangle is 7/8 meter long and 4/7 meter wide. What is its area?",
        "preferredAnswer": "1/2 square meter",
        "acceptedAnswers": [
          "1/2 square meter"
        ],
        "topic": "Area with fraction side lengths",
        "looseAcceptedAnswers": [
          "1/2 square meter",
          "0.5 square meter",
          "1/2 m^2",
          "0.5 m^2",
          "1/2 m²",
          "0.5 m²"
        ]
      },
      {
        "prompt": "You have 6 liters of juice. Each serving is 1/6 liter. How many servings can you pour?",
        "preferredAnswer": "36 servings",
        "acceptedAnswers": [
          "36 servings"
        ],
        "topic": "Divide whole number by unit fraction",
        "looseAcceptedAnswers": [
          "36 servings",
          "36 serving"
        ]
      },
      {
        "prompt": "A 5/6 cup of sugar is split equally among 10 small recipes. How much sugar does each recipe get?",
        "preferredAnswer": "1/12 cup",
        "acceptedAnswers": [
          "1/12 cup"
        ],
        "topic": "Divide fraction by whole number",
        "looseAcceptedAnswers": [
          "1/12 cup"
        ]
      },
      {
        "prompt": "Thirteen granola bars are shared equally by 6 children. How many granola bars does each child get?",
        "preferredAnswer": "2 1/6 granola bars",
        "acceptedAnswers": [
          "2 1/6 granola bars"
        ],
        "topic": "Division as fraction",
        "looseAcceptedAnswers": [
          "2 1/6 granola bars",
          "13/6 granola bars",
          "2 and 1/6 granola bars"
        ]
      },
      {
        "prompt": "How much greater is 0.91 than 7/8?",
        "preferredAnswer": "0.035",
        "acceptedAnswers": [
          "0.035"
        ],
        "topic": "Fraction/decimal conversion and subtraction",
        "looseAcceptedAnswers": [
          "0.035",
          "35/1000",
          "7/200"
        ]
      },
      {
        "prompt": "Use 1 foot = 12 inches and 1 yard = 3 feet. Add 2 yards 2 feet + 5 feet 10 inches. Give your answer in inches.",
        "preferredAnswer": "166 inches",
        "acceptedAnswers": [
          "166 inches"
        ],
        "topic": "Measurement conversion",
        "looseAcceptedAnswers": [
          "166 inches",
          "166 in",
          "166 in."
        ]
      },
      {
        "prompt": "A show starts at 4:05 PM. The first part lasts 27 minutes, and the second part lasts 1 hour 48 minutes. How many minutes after 4:05 PM does everything end?",
        "preferredAnswer": "135 minutes",
        "acceptedAnswers": [
          "135 minutes"
        ],
        "topic": "Elapsed time",
        "looseAcceptedAnswers": [
          "135 minutes",
          "135 minute",
          "135 min",
          "135"
        ]
      },
      {
        "prompt": "A rectangular garden is 31 inches long and 27 inches wide. Wes buys 10 feet of fence. If he fences the garden once around, how many inches of fence are left?",
        "preferredAnswer": "4 inches",
        "acceptedAnswers": [
          "4 inches"
        ],
        "topic": "Perimeter with unit conversion",
        "looseAcceptedAnswers": [
          "4 inches",
          "4 in",
          "4 in."
        ]
      },
      {
        "prompt": "Three adjacent angles form a straight line. Two of the angles measure 62 degrees and 49 degrees. What is the measure of the third angle?",
        "preferredAnswer": "69 degrees",
        "acceptedAnswers": [
          "69 degrees"
        ],
        "topic": "Angle measure; straight angle",
        "looseAcceptedAnswers": [
          "69 degrees",
          "69°"
        ]
      },
      {
        "prompt": "A rectangle is 23 centimeters long and 14 centimeters wide. What is its area?",
        "preferredAnswer": "322 square centimeters",
        "acceptedAnswers": [
          "322 square centimeters"
        ],
        "topic": "Area of a rectangle",
        "looseAcceptedAnswers": [
          "322 square centimeters",
          "322 sq cm",
          "322 cm^2",
          "322 cm²"
        ]
      },
      {
        "prompt": "Points A(0,3), B(0,10), C(8,10), and D(8,3) are connected in order on a graph. What is the area of the shape?",
        "preferredAnswer": "56 square units",
        "acceptedAnswers": [
          "56 square units"
        ],
        "topic": "Coordinate plane; area of rectangle",
        "looseAcceptedAnswers": [
          "56 square units",
          "56 units^2",
          "56 units²"
        ]
      },
      {
        "prompt": "A rectangular prism has length 10 cm, width 3 cm, and height 7 cm. What is its volume?",
        "preferredAnswer": "210 cubic centimeters",
        "acceptedAnswers": [
          "210 cubic centimeters"
        ],
        "topic": "Volume of rectangular prism",
        "looseAcceptedAnswers": [
          "210 cubic centimeters",
          "210 cubic cm",
          "210 cm^3",
          "210 cm³"
        ]
      },
      {
        "prompt": "A solid is made from two rectangular prisms with no overlap. Prism 1 is 7 inches by 2 inches by 5 inches. Prism 2 is 3 inches by 2 inches by 8 inches. What is the total volume?",
        "preferredAnswer": "118 cubic inches",
        "acceptedAnswers": [
          "118 cubic inches"
        ],
        "topic": "Additive volume",
        "looseAcceptedAnswers": [
          "118 cubic inches",
          "118 in^3",
          "118 in³"
        ]
      },
      {
        "prompt": "Pattern A starts at 11 and adds 5 each time. What is the 15th term of Pattern A?",
        "preferredAnswer": "81",
        "acceptedAnswers": [
          "81"
        ],
        "topic": "Numerical patterns",
        "looseAcceptedAnswers": [
          "81"
        ]
      },
      {
        "prompt": "A field trip has 105 students and 5 adults. Each bus seats 40 people. Each bus costs $295. Admission costs $6.75 per student and $12.50 per adult. The class has already raised $1,400. How much more money is needed?",
        "preferredAnswer": "$256.25",
        "acceptedAnswers": [
          "$256.25"
        ],
        "topic": "Multi-step word problem; operations with decimals",
        "looseAcceptedAnswers": [
          "$256.25",
          "256.25 dollars",
          "256.25"
        ]
      },
      {
        "prompt": "Use the line plot below. It shows ribbon lengths in inches. What is the total length of all the ribbons in inches?",
        "preferredAnswer": "5 3/4 inches",
        "acceptedAnswers": [
          "5 3/4 inches"
        ],
        "topic": "Measurement data",
        "visual": {
          "type": "linePlot",
          "title": "Line plot: ribbon lengths in inches",
          "columns": [
            {
              "label": "1/4",
              "marks": 4
            },
            {
              "label": "1/2",
              "marks": 1
            },
            {
              "label": "3/4",
              "marks": 3
            },
            {
              "label": "1",
              "marks": 2
            }
          ]
        },
        "looseAcceptedAnswers": [
          "5 3/4 inches",
          "23/4 inches",
          "5.75 inches",
          "5 and 3/4 inches",
          "5 3/4 in",
          "23/4 in",
          "5.75 in"
        ]
      }
    ]
  },
  {
    "id": "double-accelerated-practice-e",
    "title": "Practice Test E",
    "description": "Final mixed readiness set: operations, fractions, decimals, geometry, measurement, and data.",
    "questions": [
      {
        "prompt": "12 × [54 ÷ (4 + 5) + 6] - 15",
        "preferredAnswer": "129",
        "acceptedAnswers": [
          "129"
        ],
        "topic": "Order of operations",
        "looseAcceptedAnswers": [
          "129"
        ]
      },
      {
        "prompt": "(2 × 6 + 1)(3 + 2)",
        "preferredAnswer": "65",
        "acceptedAnswers": [
          "65"
        ],
        "topic": "Order of operations",
        "looseAcceptedAnswers": [
          "65"
        ]
      },
      {
        "prompt": "A school prints 629 packets. Each packet has 28 pages. How many pages are printed in all?",
        "preferredAnswer": "17,612 pages",
        "acceptedAnswers": [
          "17,612 pages"
        ],
        "topic": "Multi-digit multiplication",
        "looseAcceptedAnswers": [
          "17,612 pages",
          "17612 pages"
        ]
      },
      {
        "prompt": "9,975 ÷ 75",
        "preferredAnswer": "133",
        "acceptedAnswers": [
          "133"
        ],
        "topic": "Multi-digit division",
        "looseAcceptedAnswers": [
          "133"
        ]
      },
      {
        "prompt": "5.7 ÷ 0.01",
        "preferredAnswer": "570",
        "acceptedAnswers": [
          "570"
        ],
        "topic": "Powers of ten; decimal division",
        "looseAcceptedAnswers": [
          "570"
        ]
      },
      {
        "prompt": "In the number 18.707, the first 7 has a value of 0.7 and the second 7 has a value of 0.007. How many times as great is 0.7 as 0.007?",
        "preferredAnswer": "100",
        "acceptedAnswers": [
          "100"
        ],
        "topic": "Decimal place value comparison",
        "looseAcceptedAnswers": [
          "100"
        ]
      },
      {
        "prompt": "64.05 - 18.7 + 2.65",
        "preferredAnswer": "48",
        "acceptedAnswers": [
          "48"
        ],
        "topic": "Decimal addition and subtraction",
        "looseAcceptedAnswers": [
          "48",
          "48.0",
          "48.00"
        ]
      },
      {
        "prompt": "A 14.4-meter ribbon is cut into pieces that are each 0.6 meter long. How many pieces can be cut?",
        "preferredAnswer": "24 pieces",
        "acceptedAnswers": [
          "24 pieces"
        ],
        "topic": "Decimal division",
        "looseAcceptedAnswers": [
          "24 pieces",
          "24 piece"
        ]
      },
      {
        "prompt": "Maya buys 2 books for $6.85 each and 5 pencils for $0.75 each. She pays with a $20 bill. How much change should she get?",
        "preferredAnswer": "$2.55",
        "acceptedAnswers": [
          "$2.55"
        ],
        "topic": "Multi-step decimal money problem",
        "looseAcceptedAnswers": [
          "$2.55",
          "2.55 dollars",
          "2.55"
        ]
      },
      {
        "prompt": "What is the difference between 11/12 and 5/6?",
        "preferredAnswer": "1/12",
        "acceptedAnswers": [
          "1/12"
        ],
        "topic": "Compare/subtract fractions with unlike denominators",
        "looseAcceptedAnswers": [
          "1/12"
        ]
      },
      {
        "prompt": "4/5 + 7/10 - 3/20",
        "preferredAnswer": "1 7/20",
        "acceptedAnswers": [
          "1 7/20"
        ],
        "topic": "Add/subtract fractions with unlike denominators",
        "looseAcceptedAnswers": [
          "1 7/20",
          "27/20",
          "1.35",
          "1 and 7/20"
        ]
      },
      {
        "prompt": "9 1/3 - 4 5/6",
        "preferredAnswer": "4 1/2",
        "acceptedAnswers": [
          "4 1/2"
        ],
        "topic": "Mixed-number subtraction",
        "looseAcceptedAnswers": [
          "4 1/2",
          "9/2",
          "4.5",
          "4 and 1/2"
        ]
      },
      {
        "prompt": "One batch of trail mix uses 5/6 cup of oats and 3/4 cup of raisins. How many total cups of ingredients are needed for 4 1/2 batches?",
        "preferredAnswer": "7 1/8 cups",
        "acceptedAnswers": [
          "7 1/8 cups"
        ],
        "topic": "Fraction multiplication; scaling",
        "looseAcceptedAnswers": [
          "7 1/8 cups",
          "57/8 cups",
          "7.125 cups",
          "7 and 1/8 cups"
        ]
      },
      {
        "prompt": "7.5 × 16",
        "preferredAnswer": "120",
        "acceptedAnswers": [
          "120"
        ],
        "topic": "Whole-number-times-decimal multiplication",
        "looseAcceptedAnswers": [
          "120"
        ]
      },
      {
        "prompt": "A rectangle is 9/10 meter long and 5/9 meter wide. What is its area?",
        "preferredAnswer": "1/2 square meter",
        "acceptedAnswers": [
          "1/2 square meter"
        ],
        "topic": "Area with fraction side lengths",
        "looseAcceptedAnswers": [
          "1/2 square meter",
          "0.5 square meter",
          "1/2 m^2",
          "0.5 m^2",
          "1/2 m²",
          "0.5 m²"
        ]
      },
      {
        "prompt": "You have 8 liters of juice. Each serving is 1/4 liter. How many servings can you pour?",
        "preferredAnswer": "32 servings",
        "acceptedAnswers": [
          "32 servings"
        ],
        "topic": "Divide whole number by unit fraction",
        "looseAcceptedAnswers": [
          "32 servings",
          "32 serving"
        ]
      },
      {
        "prompt": "A 3/5 cup of sugar is split equally among 9 small recipes. How much sugar does each recipe get?",
        "preferredAnswer": "1/15 cup",
        "acceptedAnswers": [
          "1/15 cup"
        ],
        "topic": "Divide fraction by whole number",
        "looseAcceptedAnswers": [
          "1/15 cup"
        ]
      },
      {
        "prompt": "Fifteen granola bars are shared equally by 8 children. How many granola bars does each child get?",
        "preferredAnswer": "1 7/8 granola bars",
        "acceptedAnswers": [
          "1 7/8 granola bars"
        ],
        "topic": "Division as fraction",
        "looseAcceptedAnswers": [
          "1 7/8 granola bars",
          "15/8 granola bars",
          "1.875 granola bars",
          "1 and 7/8 granola bars"
        ]
      },
      {
        "prompt": "How much greater is 0.99 than 9/10?",
        "preferredAnswer": "0.09",
        "acceptedAnswers": [
          "0.09"
        ],
        "topic": "Fraction/decimal conversion and subtraction",
        "looseAcceptedAnswers": [
          "0.09",
          "9/100"
        ]
      },
      {
        "prompt": "Use 1 foot = 12 inches and 1 yard = 3 feet. Add 6 feet 7 inches + 1 yard 2 feet. Give your answer in inches.",
        "preferredAnswer": "139 inches",
        "acceptedAnswers": [
          "139 inches"
        ],
        "topic": "Measurement conversion",
        "looseAcceptedAnswers": [
          "139 inches",
          "139 in",
          "139 in."
        ]
      },
      {
        "prompt": "A movie starts at 11:30 AM. Previews last 12 minutes, and the movie lasts 2 hours 5 minutes. How many minutes after 11:30 AM does everything end?",
        "preferredAnswer": "137 minutes",
        "acceptedAnswers": [
          "137 minutes"
        ],
        "topic": "Elapsed time",
        "looseAcceptedAnswers": [
          "137 minutes",
          "137 minute",
          "137 min",
          "137"
        ]
      },
      {
        "prompt": "A rectangular garden is 48 inches long and 30 inches wide. Wes buys 14 feet of fence. If he fences the garden once around, how many inches of fence are left?",
        "preferredAnswer": "12 inches",
        "acceptedAnswers": [
          "12 inches"
        ],
        "topic": "Perimeter with unit conversion",
        "looseAcceptedAnswers": [
          "12 inches",
          "12 in",
          "12 in."
        ]
      },
      {
        "prompt": "Three adjacent angles form a straight line. Two of the angles measure 57 degrees and 76 degrees. What is the measure of the third angle?",
        "preferredAnswer": "47 degrees",
        "acceptedAnswers": [
          "47 degrees"
        ],
        "topic": "Angle measure; straight angle",
        "looseAcceptedAnswers": [
          "47 degrees",
          "47°"
        ]
      },
      {
        "prompt": "A rectangle is 19 centimeters long and 18 centimeters wide. What is its area?",
        "preferredAnswer": "342 square centimeters",
        "acceptedAnswers": [
          "342 square centimeters"
        ],
        "topic": "Area of a rectangle",
        "looseAcceptedAnswers": [
          "342 square centimeters",
          "342 sq cm",
          "342 cm^2",
          "342 cm²"
        ]
      },
      {
        "prompt": "Points A(4,1), B(4,6), C(13,6), and D(13,1) are connected in order on a graph. What is the area of the shape?",
        "preferredAnswer": "45 square units",
        "acceptedAnswers": [
          "45 square units"
        ],
        "topic": "Coordinate plane; area of rectangle",
        "looseAcceptedAnswers": [
          "45 square units",
          "45 units^2",
          "45 units²"
        ]
      },
      {
        "prompt": "A rectangular prism has length 6 cm, width 6 cm, and height 4 cm. What is its volume?",
        "preferredAnswer": "144 cubic centimeters",
        "acceptedAnswers": [
          "144 cubic centimeters"
        ],
        "topic": "Volume of rectangular prism",
        "looseAcceptedAnswers": [
          "144 cubic centimeters",
          "144 cubic cm",
          "144 cm^3",
          "144 cm³"
        ]
      },
      {
        "prompt": "A solid is made from two rectangular prisms with no overlap. Prism 1 is 8 inches by 3 inches by 3 inches. Prism 2 is 4 inches by 3 inches by 6 inches. What is the total volume?",
        "preferredAnswer": "144 cubic inches",
        "acceptedAnswers": [
          "144 cubic inches"
        ],
        "topic": "Additive volume",
        "looseAcceptedAnswers": [
          "144 cubic inches",
          "144 in^3",
          "144 in³"
        ]
      },
      {
        "prompt": "Pattern A starts at 5 and adds 8 each time. What is the 11th term of Pattern A?",
        "preferredAnswer": "85",
        "acceptedAnswers": [
          "85"
        ],
        "topic": "Numerical patterns",
        "looseAcceptedAnswers": [
          "85"
        ]
      },
      {
        "prompt": "A field trip has 88 students and 12 adults. Each bus seats 34 people. Each bus costs $320. Admission costs $8.25 per student and $10.50 per adult. The class has already raised $1,500. How much more money is needed?",
        "preferredAnswer": "$312",
        "acceptedAnswers": [
          "$312"
        ],
        "topic": "Multi-step word problem; operations with decimals",
        "looseAcceptedAnswers": [
          "$312",
          "312 dollars",
          "312"
        ]
      },
      {
        "prompt": "Use the line plot below. It shows ribbon lengths in inches. What is the total length of all the ribbons in inches?",
        "preferredAnswer": "6 1/4 inches",
        "acceptedAnswers": [
          "6 1/4 inches"
        ],
        "topic": "Measurement data",
        "visual": {
          "type": "linePlot",
          "title": "Line plot: ribbon lengths in inches",
          "columns": [
            {
              "label": "1/4",
              "marks": 1
            },
            {
              "label": "1/2",
              "marks": 5
            },
            {
              "label": "3/4",
              "marks": 2
            },
            {
              "label": "1",
              "marks": 2
            }
          ]
        },
        "looseAcceptedAnswers": [
          "6 1/4 inches",
          "25/4 inches",
          "6.25 inches",
          "6 and 1/4 inches",
          "6 1/4 in",
          "25/4 in",
          "6.25 in"
        ]
      }
    ]
  },
  {
    "id": "functionality-check",
    "title": "Functionality Check",
    "description": "Quick local test set.",
    "questions": [
      {
        "prompt": "(2 + 3)(4 + 1)",
        "preferredAnswer": "25",
        "acceptedAnswers": [
          "25"
        ],
        "topic": "Order of operations",
        "looseAcceptedAnswers": [
          "25"
        ]
      },
      {
        "prompt": "3/4 + 1/2",
        "preferredAnswer": "1 1/4",
        "acceptedAnswers": [
          "1 1/4"
        ],
        "topic": "Fraction addition",
        "looseAcceptedAnswers": [
          "1 1/4",
          "5/4",
          "1.25",
          "1 and 1/4"
        ]
      },
      {
        "prompt": "A pen costs $1.75. How much do 4 pens cost?",
        "preferredAnswer": "$7.00",
        "acceptedAnswers": [
          "$7.00"
        ],
        "topic": "Money multiplication",
        "looseAcceptedAnswers": [
          "$7.00",
          "$7",
          "7.00 dollars",
          "7 dollars",
          "7"
        ]
      },
      {
        "prompt": "Use the line plot below. It shows ribbon lengths in inches. What is the total length of all the ribbons in inches?",
        "preferredAnswer": "2 inches",
        "acceptedAnswers": [
          "2 inches"
        ],
        "topic": "Measurement data",
        "visual": {
          "type": "linePlot",
          "title": "Line plot: ribbon lengths in inches",
          "columns": [
            {
              "label": "1/4",
              "marks": 1
            },
            {
              "label": "1/2",
              "marks": 2
            },
            {
              "label": "3/4",
              "marks": 1
            },
            {
              "label": "1",
              "marks": 0
            }
          ]
        },
        "looseAcceptedAnswers": [
          "2 inches",
          "2 in",
          "2 in."
        ]
      }
    ]
  }
];


var appState = {
  view: "home",
  testId: null,
  questionIndex: 0
};

function getAppEl() {
  return document.getElementById("app");
}

function escapeHtml(value) {
  var s = value === null || typeof value === "undefined" ? "" : String(value);
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function loadState() {
  var state = { drafts: {}, results: {} };
  try {
    var raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw) {
      var parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object") {
        state.drafts = parsed.drafts || {};
        state.results = parsed.results || {};
      }
    }
  } catch (e) {}
  mergeCookieResults(state);
  return state;
}

function saveState(state) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch (e) {}
  writeCookieSummary(state);
}

function getCookieValue(name) {
  var parts = document.cookie ? document.cookie.split(";") : [];
  var prefix = name + "=";
  var i;
  for (i = 0; i < parts.length; i += 1) {
    var item = parts[i].replace(/^\s+/, "");
    if (item.indexOf(prefix) === 0) {
      return decodeURIComponent(item.substring(prefix.length));
    }
  }
  return "";
}

function mergeCookieResults(state) {
  var raw = getCookieValue(RESULT_COOKIE);
  if (!raw) return;
  try {
    var summary = JSON.parse(raw);
    var id;
    for (id in summary) {
      if (summary.hasOwnProperty(id) && !state.results[id]) {
        state.results[id] = {
          score: summary[id].score,
          total: summary[id].total,
          missedIndexes: summary[id].missedIndexes || [],
          formatOnlyIndexes: summary[id].formatOnlyIndexes || [],
          mathWrongIndexes: summary[id].mathWrongIndexes || [],
          finishedAt: summary[id].finishedAt || "",
          answers: [],
          fromCookieOnly: true
        };
      }
    }
  } catch (e) {}
}

function writeCookieSummary(state) {
  var summary = {};
  var id;
  for (id in state.results) {
    if (state.results.hasOwnProperty(id)) {
      summary[id] = {
        score: state.results[id].score,
        total: state.results[id].total,
        missedIndexes: state.results[id].missedIndexes || [],
        formatOnlyIndexes: state.results[id].formatOnlyIndexes || [],
        mathWrongIndexes: state.results[id].mathWrongIndexes || [],
        finishedAt: state.results[id].finishedAt || ""
      };
    }
  }
  try {
    document.cookie = RESULT_COOKIE + "=" + encodeURIComponent(JSON.stringify(summary)) + "; path=/; max-age=15552000; SameSite=Lax";
  } catch (e) {}
}

function findTest(testId) {
  var i;
  for (i = 0; i < TESTS.length; i += 1) {
    if (TESTS[i].id === testId) return TESTS[i];
  }
  return null;
}

function ensureDraft(state, testId) {
  var test = findTest(testId);
  var i;
  if (!test) return null;
  if (!state.drafts[testId]) {
    state.drafts[testId] = { answers: [], currentIndex: 0, startedAt: new Date().toISOString() };
  }
  if (!state.drafts[testId].answers) state.drafts[testId].answers = [];
  for (i = 0; i < test.questions.length; i += 1) {
    if (typeof state.drafts[testId].answers[i] === "undefined") state.drafts[testId].answers[i] = "";
  }
  if (typeof state.drafts[testId].currentIndex !== "number") state.drafts[testId].currentIndex = 0;
  return state.drafts[testId];
}

/*
  Grading model
  -------------
  This practice mirrors a placement test that scores binary (full credit or
  none) and is strict about answer format. There are two grading checks:

  - isCorrectAnswer (strict): the only check that scores the test. The student
    must write the canonical answer up to case and whitespace differences.
    Equivalents like "1.25" for "1 1/4", or "cm^2" for "square centimeters",
    do NOT score.

  - isMathCorrect (loose): a diagnostic-only check, used in the review screen
    to tell apart "wrong math" from "right math, wrong format". It accepts
    a wider set of equivalent forms (improper vs mixed, sq cm vs cm^2,
    plural vs singular, etc.).

  getAnswerStatus returns one of:
    "correct"     - strict match; the test gives credit
    "format-only" - math is right, format is wrong; no credit, shown for
                    diagnostic feedback so the student learns the convention
    "wrong"       - math is wrong (or answer is blank)
*/

function normalizeStrict(value) {
  var s = value === null || typeof value === "undefined" ? "" : String(value);
  s = s.toLowerCase();
  s = s.replace(/\u00a0/g, " ");
  s = s.replace(/\s+/g, " ");
  return s.replace(/^\s+|\s+$/g, "");
}

function normalizeLoose(value) {
  var s = value === null || typeof value === "undefined" ? "" : String(value);
  s = s.toLowerCase();
  s = s.replace(/\u00a0/g, " ");
  s = s.replace(/¼/g, " 1/4 ").replace(/½/g, " 1/2 ").replace(/¾/g, " 3/4 ");
  s = s.replace(/,/g, "");
  s = s.replace(/\$/g, "");
  s = s.replace(/°/g, " degree ");
  s = s.replace(/cm²/g, " square centimeter ");
  s = s.replace(/cm\^2/g, " square centimeter ");
  s = s.replace(/\bcm2\b/g, " square centimeter ");
  s = s.replace(/sq\.?\s*cm/g, " square centimeter ");
  s = s.replace(/square\s+centimeters?/g, " square centimeter ");
  s = s.replace(/m²/g, " square meter ");
  s = s.replace(/m\^2/g, " square meter ");
  s = s.replace(/\bm2\b/g, " square meter ");
  s = s.replace(/sq\.?\s*m\b/g, " square meter ");
  s = s.replace(/square\s+meters?/g, " square meter ");
  s = s.replace(/units²/g, " square unit ");
  s = s.replace(/units\^2/g, " square unit ");
  s = s.replace(/unit\^2/g, " square unit ");
  s = s.replace(/square\s+units?/g, " square unit ");
  s = s.replace(/cm³/g, " cubic centimeter ");
  s = s.replace(/cm\^3/g, " cubic centimeter ");
  s = s.replace(/\bcm3\b/g, " cubic centimeter ");
  s = s.replace(/cubic\s+centimeters?/g, " cubic centimeter ");
  s = s.replace(/in³/g, " cubic inch ");
  s = s.replace(/in\^3/g, " cubic inch ");
  s = s.replace(/\bin3\b/g, " cubic inch ");
  s = s.replace(/cubic\s+inches/g, " cubic inch ");
  s = s.replace(/inches/g, " inch ");
  s = s.replace(/\bin\./g, " inch ");
  s = s.replace(/\bin\b/g, " inch ");
  s = s.replace(/feet/g, " foot ");
  s = s.replace(/\bft\.?\b/g, " foot ");
  s = s.replace(/minutes/g, " minute ");
  s = s.replace(/\bmins\b/g, " minute ");
  s = s.replace(/\bmin\b/g, " minute ");
  s = s.replace(/degrees/g, " degree ");
  s = s.replace(/cups/g, " cup ");
  s = s.replace(/servings/g, " serving ");
  s = s.replace(/pieces/g, " piece ");
  s = s.replace(/pages/g, " page ");
  s = s.replace(/cards/g, " card ");
  s = s.replace(/stickers/g, " sticker ");
  s = s.replace(/bars/g, " bar ");
  s = s.replace(/dollars/g, " dollar ");
  s = s.replace(/\band\b/g, " ");
  s = s.replace(/\s*\/\s*/g, "/");
  s = s.replace(/[\.\?\!]+\s*$/g, "");
  s = s.replace(/(\d+\.\d*?)0+(?!\d)/g, "$1");
  s = s.replace(/(\d+)\.(?!\d)/g, "$1");
  s = s.replace(/\s+/g, " ");
  return s.replace(/^\s+|\s+$/g, "");
}

function isCorrectAnswer(userAnswer, question) {
  var normalizedUser = normalizeStrict(userAnswer);
  if (normalizedUser === "") return false;
  var accepted = question.acceptedAnswers || [];
  var i;
  for (i = 0; i < accepted.length; i += 1) {
    if (normalizedUser === normalizeStrict(accepted[i])) return true;
  }
  return false;
}

function isMathCorrect(userAnswer, question) {
  var normalizedUser = normalizeLoose(userAnswer);
  if (normalizedUser === "") return false;
  var loose = question.looseAcceptedAnswers || question.acceptedAnswers || [];
  var i;
  for (i = 0; i < loose.length; i += 1) {
    if (normalizedUser === normalizeLoose(loose[i])) return true;
  }
  return false;
}

function getAnswerStatus(userAnswer, question) {
  if (isCorrectAnswer(userAnswer, question)) return "correct";
  if (isMathCorrect(userAnswer, question)) return "format-only";
  return "wrong";
}

function countAnswered(answers) {
  var count = 0;
  var i;
  for (i = 0; i < answers.length; i += 1) {
    if (answers[i] && String(answers[i]).replace(/^\s+|\s+$/g, "") !== "") count += 1;
  }
  return count;
}

function render() {
  if (appState.view === "test") return renderQuestion();
  if (appState.view === "preFinal") return renderPreFinal();
  if (appState.view === "missed") return renderMissedReview();
  if (appState.view === "formatRules") return renderFormatRules();
  return renderHome();
}

function renderFormatRules() {
  var html = '';
  html += '<section class="card">';
  html += '<div class="row between"><h2 style="margin:0;">Answer Format Rules</h2><button class="secondary" data-action="home">Back to tests</button></div>';
  html += '<p>Scoring is binary. Each question is worth 1 point, given <strong>only when your answer matches the exact canonical form</strong>. A right value in the wrong form scores zero. This mirrors how the real placement test will be scored.</p>';

  html += '<h3>Whole numbers</h3>';
  html += '<ul>';
  html += '<li>Use commas in numbers 1,000 or larger: <code>15,042</code> &mdash; not <code>15042</code>.</li>';
  html += '<li>No trailing period or other punctuation: <code>60</code> &mdash; not <code>60.</code></li>';
  html += '</ul>';

  html += '<h3>Decimals</h3>';
  html += '<ul>';
  html += '<li>No trailing zeros: <code>14.3</code> &mdash; not <code>14.30</code>; <code>0.045</code> &mdash; not <code>0.0450</code>.</li>';
  html += '<li>Always include the leading zero: <code>0.045</code> &mdash; not <code>.045</code>.</li>';
  html += '</ul>';

  html += '<h3>Fractions and mixed numbers</h3>';
  html += '<ul>';
  html += '<li>Mixed numbers, not improper fractions: write <code>1 1/4</code>, not <code>5/4</code>.</li>';
  html += '<li>Lowest terms: <code>1/2</code>, not <code>2/4</code>.</li>';
  html += '<li>One space between whole number and fraction: <code>1 1/4</code> &mdash; not <code>1and1/4</code>, not <code>1 and 1/4</code>, not <code>1\u00BC</code>.</li>';
  html += '<li>Do not convert exact fractions to decimals: write <code>1 1/4</code>, not <code>1.25</code>.</li>';
  html += '</ul>';

  html += '<h3>Money</h3>';
  html += '<ul>';
  html += '<li>Always start with a dollar sign: <code>$1.95</code>.</li>';
  html += '<li>If the answer has cents, write two decimal places: <code>$1.40</code>, <code>$10.00</code>.</li>';
  html += '<li>If the answer is a whole-dollar total with no cents at all in the problem, write it without decimals: <code>$737</code>.</li>';
  html += '<li>If unsure which form the problem expects, look at the dollars involved &mdash; problems with cent-amounts in the prices use two decimals; problems whose totals come out to a clean whole dollar from whole-dollar inputs do not.</li>';
  html += '</ul>';

  html += '<h3>Units &mdash; when the problem asks for them</h3>';
  html += '<ul>';
  html += '<li>Spell units out. Do not use abbreviations or symbols.</li>';
  html += '<li>Length: <code>152 inches</code>, <code>12 inches</code> (always plural, since these answers are larger than 1). Do not write <code>in</code>, <code>in.</code>, or <code>152"</code>.</li>';
  html += '<li>Time (elapsed): always pure minutes. Write <code>130 minutes</code>, not <code>2 hours 10 minutes</code>, <code>2:10</code>, or <code>3:55 PM</code>.</li>';
  html += '<li>Angle: <code>69 degrees</code>, not <code>69\u00B0</code>.</li>';
  html += '<li>Area: <code>153 square centimeters</code>, <code>1/2 square meter</code>, <code>20 square units</code>. Not <code>cm\u00B2</code>, not <code>cm^2</code>, not <code>sq cm</code>.</li>';
  html += '<li>Volume: <code>120 cubic centimeters</code>, <code>112 cubic inches</code>. Not <code>cm\u00B3</code>, not <code>cm^3</code>.</li>';
  html += '<li>Singular vs plural: use the form English grammar uses. Values larger than 1 are plural (<code>12 inches</code>, <code>5 5/8 cups</code>). Values less than or equal to 1 are singular (<code>1/2 square meter</code>, <code>1/8 cup</code>).</li>';
  html += '</ul>';

  html += '<h3>Count units (when applicable)</h3>';
  html += '<ul>';
  html += '<li>Match the count language used in the problem: <code>18 pieces</code>, <code>12 servings</code>, <code>15,042 pages</code>, <code>8,132 cards</code>, <code>1 3/4 granola bars</code>.</li>';
  html += '</ul>';

  html += '<h3>What the practice does</h3>';
  html += '<p class="small subtle">If your answer is mathematically right but written in the wrong form, the practice still gives zero credit, but flags it as a <strong>Format error</strong> in the review screen so you can see the difference between &ldquo;I didn\'t know how to do this problem&rdquo; and &ldquo;I knew the answer but wrote it wrong.&rdquo; The placement test will not give that feedback. Train the habit here.</p>';

  html += '<div class="row" style="margin-top:16px;"><button class="secondary" data-action="home">Back to tests</button></div>';
  html += '</section>';
  getAppEl().innerHTML = html;
}

function renderHome() {
  var state = loadState();
  var html = '';
  var i;
  html += '<section class="card">';
  html += '<div class="row between"><h2 style="margin:0;">Answer Format Rules</h2><button data-action="formatRules">View rules</button></div>';
  html += '<p class="subtle" style="margin:8px 0 0;">Scoring is binary: full credit only when your answer matches the exact canonical form. <strong>Read these rules before taking a test.</strong></p>';
  html += '</section>';
  html += '<section class="card">';
  html += '<h2>Tests</h2>';
  html += '<p class="subtle">Pick a test. Completed tests show score and missed questions. In-progress answers are saved on this device.</p>';
  html += '<div class="test-grid">';
  for (i = 0; i < TESTS.length; i += 1) {
    html += renderTestCard(TESTS[i], state);
  }
  html += '</div>';
  html += '</section>';
  getAppEl().innerHTML = html;
}

function renderTestCard(test, state) {
  var result = state.results[test.id];
  var draft = state.drafts[test.id];
  var answered = draft && draft.answers ? countAnswered(draft.answers) : 0;
  var status = 'Not started';
  var pillClass = '';
  var actions = '';
  if (result) {
    status = 'Complete: ' + result.score + '/' + result.total;
    pillClass = ' done';
    actions += '<button data-action="reviewMissed" data-test-id="' + escapeHtml(test.id) + '">Review missed</button>';
    actions += '<button class="secondary" data-action="retake" data-test-id="' + escapeHtml(test.id) + '">Retake</button>';
  } else if (draft && answered > 0) {
    status = 'In progress: ' + answered + '/' + test.questions.length + ' answered';
    pillClass = ' warn';
    actions += '<button data-action="continueTest" data-test-id="' + escapeHtml(test.id) + '">Continue</button>';
    actions += '<button class="danger" data-action="clearDraft" data-test-id="' + escapeHtml(test.id) + '">Clear</button>';
  } else {
    actions += '<button data-action="startTest" data-test-id="' + escapeHtml(test.id) + '">Start</button>';
  }
  return '' +
    '<article class="card test-card">' +
      '<div class="row between"><h3>' + escapeHtml(test.title) + '</h3><span class="status-pill' + pillClass + '">' + escapeHtml(status) + '</span></div>' +
      '<p class="small subtle">' + escapeHtml(test.description || '') + '</p>' +
      '<p class="tiny subtle">' + test.questions.length + ' problems.</p>' +
      '<div class="row">' + actions + '</div>' +
    '</article>';
}

function currentDraftAndTest() {
  var state = loadState();
  var test = findTest(appState.testId);
  var draft = ensureDraft(state, appState.testId);
  return { state: state, test: test, draft: draft };
}

function renderQuestion() {
  var obj = currentDraftAndTest();
  var state = obj.state;
  var test = obj.test;
  var draft = obj.draft;
  var qIndex = appState.questionIndex;
  var question;
  var answered;
  var percent;
  var html;
  if (!test || !draft) return renderHome();
  if (qIndex < 0) qIndex = 0;
  if (qIndex >= test.questions.length) qIndex = test.questions.length - 1;
  appState.questionIndex = qIndex;
  draft.currentIndex = qIndex;
  saveState(state);
  question = test.questions[qIndex];
  answered = countAnswered(draft.answers);
  percent = Math.round(((qIndex + 1) / test.questions.length) * 100);

  html = '';
  html += '<section class="card">';
  html += '<div class="row between"><div><h2>' + escapeHtml(test.title) + '</h2><p class="subtle">Problem ' + (qIndex + 1) + ' of ' + test.questions.length + ' · ' + answered + ' answered</p></div><button class="secondary" data-action="home">Tests</button></div>';
  html += '<div class="progress-shell"><div class="progress-bar" style="width:' + percent + '%"></div></div>';
  html += '<div class="question-text">' + escapeHtml(question.prompt) + '</div>';
  html += renderVisual(question.visual);
  html += '<label class="answer-label" for="answerInput">Final answer</label>';
  html += '<input id="answerInput" class="answer-input" type="text" autocomplete="off" value="' + escapeHtml(draft.answers[qIndex] || '') + '">';
  html += '<div class="row" style="margin-top:16px;">';
  html += '<button class="secondary" data-action="prev"' + (qIndex === 0 ? ' disabled' : '') + '>Previous</button>';
  html += '<button data-action="next"' + (qIndex === test.questions.length - 1 ? ' disabled' : '') + '>Next</button>';
  html += '<button class="secondary" data-action="preFinal">Review answers</button>';
  html += '<button data-action="finalize">Finalize</button>';
  html += '</div>';
  html += renderNumberGrid(test, draft, qIndex);
  html += '</section>';
  getAppEl().innerHTML = html;
  bindAnswerInput();
  var input = document.getElementById('answerInput');
  if (input) input.focus();
}

function bindAnswerInput() {
  var input = document.getElementById('answerInput');
  if (!input) return;
  input.addEventListener('input', function () {
    saveCurrentAnswer(input.value);
  });
  input.addEventListener('keydown', function (event) {
    if (event.key === 'Enter') {
      saveCurrentAnswer(input.value);
      if (appState.questionIndex < findTest(appState.testId).questions.length - 1) {
        appState.questionIndex += 1;
        renderQuestion();
      }
    }
  });
}

function saveCurrentAnswer(value) {
  var state = loadState();
  var draft = ensureDraft(state, appState.testId);
  if (!draft) return;
  draft.answers[appState.questionIndex] = value;
  draft.currentIndex = appState.questionIndex;
  saveState(state);
}

function renderNumberGrid(test, draft, currentIndex) {
  var html = '<div class="number-grid" aria-label="Problem navigation">';
  var i;
  var cls;
  for (i = 0; i < test.questions.length; i += 1) {
    cls = 'secondary';
    if (draft.answers[i] && String(draft.answers[i]).replace(/^\s+|\s+$/g, '') !== '') cls += ' answered';
    if (i === currentIndex) cls += ' current';
    html += '<button class="' + cls + '" data-action="jump" data-index="' + i + '">' + (i + 1) + '</button>';
  }
  html += '</div>';
  return html;
}

function renderVisual(visual) {
  if (!visual) return '';
  if (visual.type === 'linePlot') return renderLinePlot(visual);
  if (visual.type === 'table') return renderTableVisual(visual);
  if (visual.type === 'image') return '<div class="visual-card"><img src="' + escapeHtml(visual.src) + '" alt="' + escapeHtml(visual.alt || '') + '" style="max-width:100%;height:auto;"></div>';
  return '';
}

function renderLinePlot(visual) {
  var columns = visual.columns || [];
  var maxMarks = 0;
  var i;
  var level;
  var html = '';
  for (i = 0; i < columns.length; i += 1) {
    if (columns[i].marks > maxMarks) maxMarks = columns[i].marks;
  }
  html += '<div class="visual-card">';
  html += '<div class="line-title">' + escapeHtml(visual.title || 'Line plot') + '</div>';
  html += '<table class="line-plot"><tbody>';
  for (level = maxMarks; level >= 1; level -= 1) {
    html += '<tr>';
    for (i = 0; i < columns.length; i += 1) {
      html += '<td>' + (columns[i].marks >= level ? 'x' : '&nbsp;') + '</td>';
    }
    html += '</tr>';
  }
  html += '<tr>';
  for (i = 0; i < columns.length; i += 1) {
    html += '<td>-</td>';
  }
  html += '</tr><tr>';
  for (i = 0; i < columns.length; i += 1) {
    html += '<th>' + escapeHtml(columns[i].label) + '</th>';
  }
  html += '</tr></tbody></table></div>';
  return html;
}

function renderTableVisual(visual) {
  var html = '<div class="visual-card"><table class="review-table">';
  var i;
  var j;
  if (visual.headers) {
    html += '<thead><tr>';
    for (i = 0; i < visual.headers.length; i += 1) html += '<th>' + escapeHtml(visual.headers[i]) + '</th>';
    html += '</tr></thead>';
  }
  html += '<tbody>';
  for (i = 0; i < (visual.rows || []).length; i += 1) {
    html += '<tr>';
    for (j = 0; j < visual.rows[i].length; j += 1) html += '<td>' + escapeHtml(visual.rows[i][j]) + '</td>';
    html += '</tr>';
  }
  html += '</tbody></table></div>';
  return html;
}

function renderPreFinal() {
  var obj = currentDraftAndTest();
  var test = obj.test;
  var draft = obj.draft;
  var html = '';
  var i;
  if (!test || !draft) return renderHome();
  html += '<section class="card">';
  html += '<div class="row between"><div><h2>Review answers</h2><p class="subtle">' + escapeHtml(test.title) + '</p></div><button class="secondary" data-action="backToQuestion">Back</button></div>';
  html += '<table class="review-table"><thead><tr><th>#</th><th>Your answer</th><th>Go back</th></tr></thead><tbody>';
  for (i = 0; i < test.questions.length; i += 1) {
    html += '<tr><td>' + (i + 1) + '</td><td>' + escapeHtml(draft.answers[i] || '') + '</td><td><button class="secondary" data-action="jump" data-index="' + i + '">Edit</button></td></tr>';
  }
  html += '</tbody></table>';
  html += '<div class="row" style="margin-top:16px;"><button data-action="finalize">Finalize test</button><button class="secondary" data-action="backToQuestion">Back to current problem</button></div>';
  html += '</section>';
  getAppEl().innerHTML = html;
}

function finalizeTest() {
  var state = loadState();
  var test = findTest(appState.testId);
  var draft = ensureDraft(state, appState.testId);
  var unanswered;
  var i;
  var status;
  var missed = [];
  var formatOnly = [];
  var mathWrong = [];
  var score = 0;
  if (!test || !draft) return;
  unanswered = test.questions.length - countAnswered(draft.answers);
  if (unanswered > 0) {
    if (!window.confirm('There are ' + unanswered + ' unanswered problems. Finalize anyway?')) return;
  } else {
    if (!window.confirm('Finalize this test and score it now?')) return;
  }
  for (i = 0; i < test.questions.length; i += 1) {
    status = getAnswerStatus(draft.answers[i], test.questions[i]);
    if (status === "correct") {
      score += 1;
    } else {
      missed.push(i);
      if (status === "format-only") formatOnly.push(i);
      else mathWrong.push(i);
    }
  }
  state.results[test.id] = {
    score: score,
    total: test.questions.length,
    missedIndexes: missed,
    formatOnlyIndexes: formatOnly,
    mathWrongIndexes: mathWrong,
    answers: draft.answers.slice(0),
    finishedAt: new Date().toISOString()
  };
  delete state.drafts[test.id];
  saveState(state);
  appState.view = 'missed';
  render();
}

function renderMissedReview() {
  var state = loadState();
  var test = findTest(appState.testId);
  var result = test ? state.results[test.id] : null;
  var missed;
  var formatOnly;
  var mathWrong;
  var formatSet;
  var html = '';
  var i;
  var idx;
  var question;
  var userAnswer;
  var isFormatOnly;
  var statusLabel;
  var statusClass;
  if (!test || !result) return renderHome();
  missed = result.missedIndexes || [];
  formatOnly = result.formatOnlyIndexes || [];
  mathWrong = result.mathWrongIndexes || [];
  formatSet = {};
  for (i = 0; i < formatOnly.length; i += 1) formatSet[formatOnly[i]] = true;
  html += '<section class="card">';
  html += '<div class="row between"><div><h2>' + escapeHtml(test.title) + '</h2><p class="subtle">Score: ' + result.score + '/' + result.total + '</p></div><button class="secondary" data-action="home">Tests</button></div>';
  if (missed.length === 0) {
    html += '<div class="notice">Perfect score. No missed problems.</div>';
  } else {
    html += '<div class="notice">';
    html += '<p style="margin-top:0"><strong>Scoring is binary:</strong> the placement test gives full credit only for the exact canonical form. Equivalent values in a wrong format (improper fraction instead of mixed number, missing units, etc.) score zero on the real test, so they score zero here too.</p>';
    html += '<p style="margin-bottom:0">Missed problems below are tagged either <strong>Math error</strong> (wrong value) or <strong>Format error</strong> (right value, wrong form). Format errors are the most important to fix &mdash; the math is right, the habit isn\'t.</p>';
    html += '</div>';
    html += '<p class="small"><strong>Summary:</strong> ' + mathWrong.length + ' math error' + (mathWrong.length === 1 ? '' : 's') + ', ' + formatOnly.length + ' format error' + (formatOnly.length === 1 ? '' : 's') + '.';
    html += ' <button class="secondary" data-action="formatRules" style="margin-left:8px;">See answer format rules</button></p>';
    for (i = 0; i < missed.length; i += 1) {
      idx = missed[i];
      question = test.questions[idx];
      userAnswer = result.answers && typeof result.answers[idx] !== "undefined" ? result.answers[idx] : '';
      isFormatOnly = !!formatSet[idx];
      statusLabel = isFormatOnly ? 'Format error' : 'Math error';
      statusClass = isFormatOnly ? 'warn' : '';
      html += '<div class="card">';
      html += '<div class="row between"><h3 style="margin:0;">Problem ' + (idx + 1) + '</h3><span class="status-pill ' + statusClass + '">' + statusLabel + '</span></div>';
      html += '<div class="question-text">' + escapeHtml(question.prompt) + '</div>';
      html += renderVisual(question.visual);
      html += '<p><strong>Your answer:</strong> <span class="incorrect">' + escapeHtml(userAnswer === '' ? '(blank)' : userAnswer) + '</span></p>';
      html += '<p><strong>Correct answer:</strong> <span class="correct">' + escapeHtml(question.preferredAnswer) + '</span></p>';
      if (isFormatOnly) {
        html += '<p class="small subtle"><em>The math is right but the form doesn\'t match what the placement test will accept. Practice writing the exact form shown above.</em></p>';
      }
      if (question.topic) html += '<p class="small subtle"><strong>Topic:</strong> ' + escapeHtml(question.topic) + '</p>';
      html += '</div>';
    }
  }
  html += '<div class="row"><button class="secondary" data-action="home">Back to tests</button><button class="secondary" data-action="retake" data-test-id="' + escapeHtml(test.id) + '">Retake this test</button></div>';
  html += '</section>';
  getAppEl().innerHTML = html;
}

function startTest(testId, reset) {
  var state = loadState();
  if (reset) {
    delete state.results[testId];
    delete state.drafts[testId];
  }
  ensureDraft(state, testId);
  saveState(state);
  appState.view = 'test';
  appState.testId = testId;
  appState.questionIndex = state.drafts[testId].currentIndex || 0;
  render();
}

function clearDraft(testId) {
  var state = loadState();
  if (window.confirm('Clear saved answers for this in-progress test?')) {
    delete state.drafts[testId];
    saveState(state);
    renderHome();
  }
}

function retake(testId) {
  if (!window.confirm('Retake this test? This will remove the saved score for this test on this device.')) return;
  startTest(testId, true);
}

function handleClick(event) {
  var target = event.target;
  var action;
  var testId;
  var index;
  while (target && target !== document && !target.getAttribute('data-action')) target = target.parentNode;
  if (!target || target === document) return;
  action = target.getAttribute('data-action');
  testId = target.getAttribute('data-test-id');
  if (action === 'home') {
    appState.view = 'home';
    render();
  } else if (action === 'startTest') {
    startTest(testId, false);
  } else if (action === 'continueTest') {
    startTest(testId, false);
  } else if (action === 'clearDraft') {
    clearDraft(testId);
  } else if (action === 'retake') {
    retake(testId);
  } else if (action === 'reviewMissed') {
    appState.view = 'missed';
    appState.testId = testId;
    render();
  } else if (action === 'prev') {
    saveFromInputThenMove(-1);
  } else if (action === 'next') {
    saveFromInputThenMove(1);
  } else if (action === 'jump') {
    index = parseInt(target.getAttribute('data-index'), 10);
    if (!isNaN(index)) {
      saveCurrentVisibleInput();
      appState.view = 'test';
      appState.questionIndex = index;
      render();
    }
  } else if (action === 'preFinal') {
    saveCurrentVisibleInput();
    appState.view = 'preFinal';
    render();
  } else if (action === 'backToQuestion') {
    appState.view = 'test';
    render();
  } else if (action === 'finalize') {
    saveCurrentVisibleInput();
    finalizeTest();
  } else if (action === 'formatRules') {
    appState.view = 'formatRules';
    render();
  }
}

function saveCurrentVisibleInput() {
  var input = document.getElementById('answerInput');
  if (input) saveCurrentAnswer(input.value);
}

function saveFromInputThenMove(delta) {
  var test = findTest(appState.testId);
  if (!test) return;
  saveCurrentVisibleInput();
  appState.questionIndex += delta;
  if (appState.questionIndex < 0) appState.questionIndex = 0;
  if (appState.questionIndex >= test.questions.length) appState.questionIndex = test.questions.length - 1;
  renderQuestion();
}

function hasAdjacentParentheses(text) {
  return /\)\s*\(/.test(text || "");
}

function runSelfCheck() {
  var errors = [];
  var t;
  var q;
  var test;
  var question;
  var linePlots;
  var adjacentGroups;
  var expectedLength;
  for (t = 0; t < TESTS.length; t += 1) {
    test = TESTS[t];
    expectedLength = test.id === "functionality-check" ? 4 : 30;
    if (!test.questions || test.questions.length !== expectedLength) errors.push(test.title + ' has the wrong number of questions.');
    linePlots = 0;
    adjacentGroups = 0;
    for (q = 0; q < test.questions.length; q += 1) {
      question = test.questions[q];
      if (question.visual && question.visual.type === 'linePlot') linePlots += 1;
      if (hasAdjacentParentheses(question.prompt)) adjacentGroups += 1;
      if (!isCorrectAnswer(question.preferredAnswer, question)) errors.push(test.title + ' problem ' + (q + 1) + ' preferred answer fails scoring.');
      if (question.prompt.indexOf('Com' + 'pute:') >= 0 || question.prompt.indexOf('Eval' + 'uate:') >= 0) errors.push(test.title + ' problem ' + (q + 1) + ' contains compute/evaluate prefix.');
    }
    if (linePlots < 1) errors.push(test.title + ' is missing a required visual item.');
    if (adjacentGroups < 1) errors.push(test.title + ' is missing a required grouped-expression item.');
  }
  if (errors.length && window.console && window.console.error) window.console.error('Self-check errors:', errors);
  return errors;
}

function init() {
  runSelfCheck();
  document.addEventListener('click', handleClick);
  renderHome();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
