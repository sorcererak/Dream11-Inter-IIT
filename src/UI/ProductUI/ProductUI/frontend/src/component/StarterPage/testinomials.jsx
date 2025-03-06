import TestimonialCard from "./TestinomialCard";


const testimonials = [
  {
    avatar: "https://cdnflags.dream11.com/d11-static-pages/images/manasMalhotra.webp",
    name: "Manas Malhotra",
    role: "MI vs RR: 1 Crore Winnings",
    rating: 5,
    content: "I've been playing on Dream11 for many years and I have won earlier too. This time I won a Mega Contest! I'm an avid cricket follower. The best part was I got my winnings instantly into my bank account after the withdrawal.",
  },
  {
    avatar: "https://cdnflags.dream11.com/d11-static-pages/images/sagarBhagat.webp",
    name: "Sagar Bhagat",
    role: "PBKS vs DC: 1 Crore Winnings",
    rating: 5,
    content: "One of the many reasons why I trust Dream11 is because I get my winnings safely and instantly after the contest gets over. I also used my own cricket knowledge to enhance my fantasy cricket skills.",
  },
  {
    avatar: "https://cdnflags.dream11.com/d11-static-pages/images/AnujYadav.webp",
    name: "Anuj Yadav",
    role: "GT vs MI: 1 Crore Winnings",
    rating: 5,
    content: "I used to always watch cricket and analyse the game. It was all about doing some good research and putting skills to work. I'm glad my knowledge and skills got rewarded and I won the Mega Contest.",
  }
];

const Testimonial = () => {
  return (
    <section className="py-12 bg-gray-50 sm:py-16 lg:py-20" id="winners">
      <div className="px-4 mx-auto max-w-7xl sm:px-6 lg:px-8">
        <div className="text-center">
          <p className="text-lg font-medium text-gray-600">Ratings and Reviews</p>
          <h2 className="mt-4 text-3xl font-bold text-gray-900 sm:text-4xl xl:text-5xl">Our Happy Users Say About Us</h2>
        </div>
        <div className="grid grid-cols-1 gap-6 mt-10 md:grid-cols-3">
          {testimonials.map((testimonial, index) => (
            <TestimonialCard key={index} {...testimonial} />
          ))}
        </div>
        {/* <div className="mt-8 text-center">
          <a href="#" className="text-base font-bold text-gray-900 border-b-2 border-gray-900 hover:border-gray-600">
            Check all 2,157 reviews
          </a>
        </div> */}
      </div>
    </section>
  );
};

export default Testimonial;
