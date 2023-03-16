print('Start creating database');


db.createUser(
    {
        user: "root",
        pwd: "password",
        roles: [
            {
                role: "readWrite",
                db: "mlflowexperiments"
            }
        ]
    }
);

