import { useState } from "react";
import { Accordion, AccordionSummary, AccordionDetails, Button, Box, Typography } from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import styles from '../../css/faqs.module.css'
import shootmail from '../../assets/landing_page/shootmail.svg'

const FAQ = () => {
    const [expanded, setExpanded] = useState(false);

    const handleChange = (panel) => (_, isExpanded) => {
        setExpanded(isExpanded ? panel : false);
    };

    return (
        <Box
            sx={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "flex-start",
                backgroundColor: "#f9f9f9",
                padding: "2rem",
                gap: "2rem",
                borderRadius: "8px",
            }}
            id="faqs"
        >
            {/* FAQ Section */}
            <Box sx={{ flex: 2 }}>
                <div className={styles.heading}>
                    FREQUENTLY <br />ASKED QUESTIONS
                </div>

                {/* FAQ Items */}
                {faqData.map((faq, index) => (
                    <Accordion
                        key={index}
                        expanded={expanded === `panel${index}`}
                        onChange={handleChange(`panel${index}`)}
                        sx={{
                            boxShadow: "none",
                            border: "1px solid #ddd",
                            borderRadius: "8px",
                            marginBottom: "1rem",
                            "&:before": { display: "none" }, // Remove default accordion line
                        }}
                    >
                        <AccordionSummary
                            expandIcon={<ExpandMoreIcon />}
                            aria-controls={`panel${index}-content`}
                            id={`panel${index}-header`}
                            sx={{
                                fontWeight: 600,
                                fontSize: "1rem",
                                color: "#333",
                            }}
                        >
                            {faq.question}
                        </AccordionSummary>
                        <AccordionDetails sx={{ fontSize: "0.9rem", color: "#666" }}>
                            {faq.answer}
                        </AccordionDetails>
                    </Accordion>
                ))}
            </Box>

            {/* Contact Section */}
            <Box
                sx={{
                    flex: 0.5,
                    padding: "1.5rem",
                    backgroundColor: "white",
                    border: "1px solid #ddd",
                    borderRadius: "8px",
                    marginTop: "9.5rem",
                    textAlign: "center",
                    boxShadow: "0 4px 8px rgba(0,0,0,0.1)",
                }}
            >
                <div className={styles.shootmailicon}><img className={styles.shootmail} src={shootmail}></img></div>
                {/* </Box> */}
                <Typography
                    variant="h6"
                    sx={{
                        fontWeight: "bold",
                        fontSize: "1.2rem",
                        marginBottom: "0.5rem",
                        color: "#333",
                    }}
                >
                    Do you have more questions?
                </Typography>
                <Typography
                    sx={{
                        fontSize: "0.9rem",
                        color: "#666",
                        marginBottom: "2rem",
                    }}
                >
                    End-to-end payments and financial management in a single solution.
                    Meet the right platform to help realize.
                </Typography>
                <Button
                    variant="contained"
                    sx={{
                        backgroundColor: "#c00",
                        color: "white",
                        textTransform: "none",
                        fontWeight: "bold",
                        "&:hover": {
                            backgroundColor: "#a00",
                        },
                    }}
                >
                    Shoot a Direct Mail
                </Button>
            </Box>

        </Box>
    );
};

// Sample FAQ Data
const faqData = [
    {
        question: "How are Dream11 points calculated?",
        answer: "Dream11 points are calculated based on player performance in actual matches. Points are awarded for actions like runs, wickets, catches, and more. For a detailed breakdown of the scoring system, visit the Dream11 Fantasy Points System page for various sports."
    },
    {
        question: "How does Dream11 work for cricket matches?",
        answer: "Dream11 allows users to create a fantasy cricket team by selecting real players from upcoming matches. Your team's performance is based on the actual performance of the players in the match. Points are awarded for actions like batting, bowling, fielding, and captaincy."
    },
    {
        question: "How does the ML model select players for the Dream Team?",
        answer: "The ML model uses advanced algorithms to evaluate player statistics, pitch conditions, weather, opposition strength, and historical data. It selects players who are likely to perform well based on these factors, optimizing the team to maximize points."
    },
    {
        question: "Does the ML Dream Team guarantee a win?",
        answer: "No, the ML Dream Team does not guarantee a win. While the team is optimized for maximum points based on data analysis, the actual performance of players during a match can vary, and real-time factors like injuries or weather can affect outcomes."
    },
    {
        question: "Does the ML model consider match formats (T20, ODI, Test)?",
        answer: "Yes, the ML model is tailored for specific formats. It adjusts: Player preferences based on match duration (e.g., explosive batsmen for T20, consistent performers for Tests) and strategy based on the scoring system for each format."
    },
    {
        question: "Can I customize the ML-generated Dream Team?",
        answer: "Yes, Dream11 allows customization of the ML-generated Dream Team. If you're not satisfied with the generated team, you can swap players by deleting them or selecting alternatives from your squad."
    },
    {
        question: "How do I know the ML model is unbiased?",
        answer: "The ML model is based on data-driven logic and statistical algorithms, not influenced by subjective opinions or fan biases. It ensures a fair and objective selection of players based on performance metrics and match conditions."
    },
    {
        question: "How does the custom match feature work on Dream11?",
        answer: "The custom match feature lets you create your own match by selecting 22 players manually or importing them from a CSV file. You can customize the match settings and generate your team based on your preferences."
    },
    {
        question: "What languages does the audio feature offer?",
        answer: "The audio feature supports languages including Hindi, English, Marathi, Tamil, Telugu, and several regional languages for accessibility."
    },
    {
        question: "How do I create my own Dream11 match?",
        answer: "To create your own Dream11 match, go to the 'Select Match' or 'Create Match' section. You can select teams, set match parameters, and choose players from the available pool. Once the match is created, you can finalize your Dream11 squad."
    },
    {
        question: "Can I import my team from an external source?",
        answer: "Yes, you can import your Dream11 team from a CSV file. This feature allows you to select players easily by uploading your pre-created squad, making team management faster and more efficient."
    },
];

export default FAQ;
