const TestimonialCard = ({ avatar, name, role, rating, content }) => {
    return (
      <div className="flex flex-col overflow-hidden shadow-xl">
        <div className="flex flex-col justify-between flex-1 p-6 bg-white lg:py-8 lg:px-7">
          <div className="flex-1">
            <div className="flex items-center">
              {Array.from({ length: 5 }).map((_, index) => (
                <svg
                  key={index}
                  className={`w-5 h-5 ${index < rating ? "text-[#FDB241]" : "text-gray-300"}`}
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                </svg>
              ))}
            </div>
            <blockquote className="flex-1 mt-6">
              <p className="text-lg leading-relaxed text-gray-900">{content}</p>
            </blockquote>
          </div>
          <div className="flex items-center mt-4">
            <img className="flex-shrink-0 object-cover rounded-full w-11 h-11" src={avatar} alt={name} />
            <div className="ml-4">
              <p className="text-base font-bold text-gray-900">{name}</p>
              <p className="mt-0.5 text-sm text-gray-500">{role}</p>
            </div>
          </div>
        </div>
      </div>
    );
  };
  
  export default TestimonialCard;
  