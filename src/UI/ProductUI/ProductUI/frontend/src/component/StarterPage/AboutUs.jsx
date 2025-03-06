import styles from "../../css/AboutUs.module.css";
import shadow from "../../assets/landing_page/aboutus.svg";
import img1 from "../../assets/landing_page/aboutus1.png";
import img2 from "../../assets/landing_page/aboutus2.png";
import img3 from "../../assets/landing_page/aboutus3.png";

const AboutUs = () => {
    return (
        <div className={styles.aboutUsContainer} id="about">
            <div className={styles.imageGrid}>
                <div className={styles.imageBox}>
                    <img src={img1} alt="Player 1" className={styles.image1} />
                    <img src={img3} alt="Player 2" className={styles.image3} />

                </div>
                <div className={`${styles.imageBox} ${styles.overlayImageBox}`}>
                    <img src={img2} alt="Team Celebration" className={styles.image2} />
                    <div className={styles.overlayText}>Over 20 Crore+ Users</div>
                </div>
            </div>

            <div className={styles.aboutContent}>
                <img src={shadow} alt="Shadow" className={styles.shadow} />
                <div className={styles.aboutbox}>
                    <h2 className={styles.title}>About us</h2>
                    <p className={styles.description}>
                        Dream11, the world’s largest fantasy sports platform, brings together over 220 million passionate sports fans. 
                        Our innovative AI-powered team selection tool, Dynamic Ranking Ensemble for Accurate Modelling (D.R.E.A.M), is a game-changer for fantasy cricket enthusiasts. This advanced machine learning solution uses cutting-edge algorithms to analyze vast amounts of data, providing accurate player performance predictions. By considering historical statistics, match conditions, player form, and other dynamic factors, D.R.E.A.M empowers users to create highly optimized fantasy cricket teams.
                        Through an intuitive interface, D.R.E.A.M not only assists users in selecting the best players but also provides real-time insights, recommendations, and strategies. The tool’s Generative AI-driven aids offer actionable guidance, helping users make well-informed team-building decisions. Whether you're a seasoned fantasy sports expert or a beginner, D.R.E.A.M is designed to elevate your fantasy sports experience, making it more accurate, data-driven, and rewarding.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default AboutUs;
