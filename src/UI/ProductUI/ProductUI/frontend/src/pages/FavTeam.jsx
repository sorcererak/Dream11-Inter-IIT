import { useState } from "react";
// import bg from "../Starterpage/bg.png";
import bg from "../assets/landing_page/landing.png";
import Slider from "react-slick";
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";
// import "./FavTeam.module.css"
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos';
import ArrowBackIosIcon from '@mui/icons-material/ArrowBackIos';

const Teams = [
    {
        id: 1,
        name: "India",
        logo: "https://upload.wikimedia.org/wikipedia/en/thumb/8/8d/Cricket_India_Crest.svg/800px-Cricket_India_Crest.svg.png",
    },
    {
        id: 2,
        name: "Australia",
        logo: "https://upload.wikimedia.org/wikipedia/en/3/3f/Cricket_Australia.png",
    },
    {
        id: 3,
        name: "India",
        logo: "https://upload.wikimedia.org/wikipedia/en/thumb/8/8d/Cricket_India_Crest.svg/800px-Cricket_India_Crest.svg.png",
    },
    {
        id: 4,
        name: "Australia",
        logo: "https://upload.wikimedia.org/wikipedia/en/3/3f/Cricket_Australia.png",
    },
    {
        id: 5,
        name: "India",
        logo: "https://upload.wikimedia.org/wikipedia/en/thumb/8/8d/Cricket_India_Crest.svg/800px-Cricket_India_Crest.svg.png",
    },
    {
        id: 6,
        name: "Australia",
        logo: "https://upload.wikimedia.org/wikipedia/en/3/3f/Cricket_Australia.png",
    },
    {
        id: 7,
        name: "India",
        logo: "https://upload.wikimedia.org/wikipedia/en/thumb/8/8d/Cricket_India_Crest.svg/800px-Cricket_India_Crest.svg.png",
    },
    {
        id: 8,
        name: "Australia",
        logo: "https://upload.wikimedia.org/wikipedia/en/3/3f/Cricket_Australia.png",
    },
    {
        id: 9,
        name: "India",
        logo: "https://upload.wikimedia.org/wikipedia/en/thumb/8/8d/Cricket_India_Crest.svg/800px-Cricket_India_Crest.svg.png",
    },
    {
        id: 10,
        name: "Australia",
        logo: "https://upload.wikimedia.org/wikipedia/en/3/3f/Cricket_Australia.png",
    },
    {
        id: 11,
        name: "India",
        logo: "https://upload.wikimedia.org/wikipedia/en/thumb/8/8d/Cricket_India_Crest.svg/800px-Cricket_India_Crest.svg.png",
    },
    {
        id: 12,
        name: "Australia",
        logo: "https://upload.wikimedia.org/wikipedia/en/3/3f/Cricket_Australia.png",
    },
];


function NextBtn(props) {
    const { className, onClick } = props;
    return (
        <div
            className={className}
            onClick={onClick}
        >
            <ArrowForwardIosIcon style={{ color: 'blue', fontSize: '30px' }} />
        </div>
    );
}

function PreviousBtn(props) {
    const { className, onClick } = props;
    return (
        <div
            className={className}
            onClick={onClick}
        >
            <div className=" text-sky-600  h-12 flex items-center justify-center w-8">
                <ArrowBackIosIcon style={{ color: 'blue', fontSize: '30px' }} />
            </div>

        </div>
    );
}
const FavTeam = () => {
    const [selectedTeams, setSelectedTeams] = useState([]);

    const handleCardClick = (id) => {
        setSelectedTeams((prevSelected) =>
            prevSelected.includes(id)
                ? prevSelected.filter((teamId) => teamId !== id) // Remove if already selected
                : [...prevSelected, id] // Add if not selected
        );
    };
    const settings = {
        // className: "center",
        // centerMode: true,
        infinite: false,
        slidesToShow: 4,
        speed: 500,
        rows: 2,
        slidesPerRow: 1,
        arrows: true,
        nextArrow: <NextBtn />,
        prevArrow: <PreviousBtn />,
    };

    return (
        <div
            className="w-screen h-screen flex justify-center items-center "
            style={{
                backgroundImage: `url(${bg})`,
                backgroundSize: "cover",
                backgroundPosition: "center",
                backgroundRepeat: "no-repeat",
            }}
        >
            <div className="bg-white w-2/3 h-[75%] rounded-lg shadow-lg p-8 bg-opacity-75">
                <h1 className="font-bold text-[#262626] text-4xl">What Are Your Favourite Teams?</h1>
                <p className="text-2xl text-gray-300 mt-2 mb-4">Select teams to customize the home page</p>
                <div className="w-1/3">
                    <NameInput />
                </div>
                <div className="mx-auto h-2/3 scale-90">
                    <Slider {...settings}>
                        {Teams.map((team) => (
                            <div
                                key={team.id}
                                className="relative flex flex-col items-center cursor-pointer"
                            >
                                <div
                                    onClick={() => handleCardClick(team.id)}
                                    className={`rounded-full h-32 w-32 flex items-center justify-center mx-auto ${selectedTeams.includes(team.id) ? "ring-4 ring-blue-500" : ""
                                        }`}
                                >
                                    <div className="rounded-full  h-32 w-32 flex items-center justify-center">
                                        <img
                                            src={team.logo}
                                            alt={team.name}
                                            className="h-24 w-24 object-contain rounded-full"
                                        />
                                    </div>
                                </div>
                                <p className="text-center font-bold mt-2">{team.name}</p>
                            </div>
                        ))}
                    </Slider>

                </div>
                <div className="flex justify-end gap-4">
                    <button className="bg-dream11 text-white px-8 py-1  rounded-none mt-4">Skip</button>
                    <button className="bg-dream11 text-white px-8 py-1 rounded-none mt-4">Save</button>
                </div>
            </div>
        </div>
    );
};

export default FavTeam;

const NameInput = () => {
    return (
        <>
            {/* <label className="mb-[10px] block text-base font-medium text-dark dark:text-white">
        Search Team
      </label> */}
            <div className="relative">
                <input
                    type="text"
                    placeholder="Search Team"
                    className="w-full bg-transparent rounded-md border border-stroke dark:border-dark-3 py-[10px] pr-3 pl-12 text-dark-6 outline-none transition focus:border-primary active:border-primary disabled:cursor-default disabled:bg-gray-2"
                />
                <span className="absolute top-1/2 left-4 -translate-y-1/2">
                    <svg
                        width={20}
                        height={20}
                        viewBox="0 0 20 20"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <path
                            d="M3.72 12.886a4.167 4.167 0 0 1 2.947-1.22h6.666a4.167 4.167 0 0 1 4.167 4.167v1.666a.833.833 0 1 1-1.667 0v-1.666a2.5 2.5 0 0 0-2.5-2.5H6.667a2.5 2.5 0 0 0-2.5 2.5v1.666a.833.833 0 1 1-1.667 0v-1.666a4.17 4.17 0 0 1 1.22-2.947ZM10 3.333a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5Zm-4.166 2.5a4.167 4.167 0 1 1 8.333 0 4.167 4.167 0 0 1-8.333 0Z"
                            opacity={0.8}
                            fillRule="evenodd"
                            clipRule="evenodd"
                            fill="#9CA3AF"
                        />
                    </svg>
                </span>
            </div>
        </>
    );
};
